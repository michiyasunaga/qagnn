from utils.data_utils import *
from torch_geometric.nn import MessagePassing
from transformers import AutoModel
from utils.layers import *

class GSCLayer(MessagePassing):
    def __init__(self):
        super(GSCLayer, self).__init__(aggr="add")

    def forward(self, x, edge_index, edge_embeddings):
        aggr_out = self.propagate(edge_index, x=(x, x), edge_attr=edge_embeddings)
        return aggr_out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr



class GSC_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, hidden_size):
        super().__init__()
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.hidden_size = hidden_size
        self.edge_encoder = nn.Sequential(nn.Embedding(num_embeddings=self.n_etype * self.n_ntype * self.n_ntype, embedding_dim=hidden_size),
                                          nn.LayerNorm(self.hidden_size), nn.GELU(),
                                          nn.Linear(in_features=hidden_size, out_features=1),
                                          nn.Sigmoid())
        self.k = k
        self.gnn_layers = nn.ModuleList([GSCLayer() for _ in range(k)])
        self.regulator = nn.Sequential(nn.Linear(1, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.GELU(),
                                       nn.Linear(self.hidden_size, 1))



    def edge_embeddings(self, edge_type, edge_idx, node_type_idxs):
        """
        In total we can have n_ntype * n_ntype * n_etype different edges. In fact, each edge is caracterized by its head, its tail node and its edge node.
        """
        node_type = node_type_idxs.view(-1).contiguous()
        head_type = node_type[edge_idx[0]]
        tail_type = node_type[edge_idx[1]]

        idx1 = self.n_ntype*head_type + tail_type
        edge_to_idx = self.n_etype*idx1 + edge_type
        edge_embeddings = self.edge_encoder(edge_to_idx)
        return edge_embeddings


    def forward(self, adj, node_type_idxs):
        """
        Understand if we need or not the regulator
        """
        _batch_size, _n_nodes = node_type_idxs.size()
        n_node_total = _batch_size * _n_nodes
        edge_idx, edge_type = adj

        edge_embeddings = self.edge_embeddings(edge_type=edge_type, edge_idx=edge_idx,  node_type_idxs=node_type_idxs)
        out = torch.zeros(n_node_total, 1).to(node_type_idxs.device)
        for i in range(self.k):
            # propagate and aggregate between nodes and edges
            out = self.gnn_layers[i](out, edge_idx, edge_embeddings)
        out = self.regulator(out).view(_batch_size, _n_nodes, -1)  # just for normalizing output
        return out




class GSC(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, question_dim,
                 gnn_dim, fc_dim, n_fc_layer, p_fc):
        super().__init__()
        self.gnn = GSC_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                         hidden_size=gnn_dim)
        self.linear_layer = MLP(question_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, question, node_type_ids,  adj):
        """
        question: (batch, dim_question)
        adj: edge_index, edge_type
        node_type_ids: (batch_size, n_node), 0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        """
        gnn_output = self.gnn(adj, node_type_ids)
        graph_score = gnn_output[:,0]
        ctx_score = self.linear_layer(question)
        return ctx_score + graph_score


class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_name.split('-')[0]
        model_class = AutoModel
        self.module = model_class.from_pretrained(self.model_name, output_hidden_states=True)
        self.question_dim = self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        input_ids, attention_mask, token_type_ids, output_mask = inputs
        outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        questions = self.module.pooler(hidden_states)
        return questions, all_hidden_states


class Model(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype,
                 gnn_dim, fc_dim, n_fc_layer, p_fc):
        super().__init__()
        self.encoder = TextEncoder(model_name)
        self.decoder = GSC(args, k, n_ntype, n_etype, self.encoder.question_dim,
                             gnn_dim, fc_dim, n_fc_layer, p_fc)

    def exctract(self, *inputs):
        # Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        inputs1 = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]]
        inputs2 = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]]
        inputs3 = [sum(x, []) for x in inputs[-2:]]
        inp = inputs1 + inputs2 + inputs3
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = inp
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device),
               edge_type.to(node_type_ids.device))

        return *lm_inputs, edge_index_orig, edge_type_orig, adj, node_type_ids, concept_ids

    def batch_graph(self, start_edge_idx, start_edge_type, n_nodes):
        #start_edge_idx list of (n, (2,E))
        #edge_type:  list of (n, (2,E))
        n = len(start_edge_idx)
        edge_idx = torch.cat([start_edge_idx[j] + j * n_nodes for j in range(n)], dim=1)
        edge_type = torch.cat(start_edge_type, dim=0)
        return edge_idx, edge_type

    def forward(self, *inputs, layer_id=-1, detail=False):

        bs, nc = inputs[0].size(0), inputs[0].size(1)
        *lm_inputs, edge_index_orig, edge_type_orig, adj, node_type_ids, concept_ids = self.exctract(*inputs)
        question, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits = self.decoder(question.to(node_type_ids.device), node_type_ids, adj)
        logits = logits.view(bs, nc)
        if not detail:
            return logits
        else:
            return logits, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            #edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            #edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )

class GSCLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        model_type = model_name.split('-')[0]
        print('train_statement_path', train_statement_path)
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print ('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)

        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)


        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)

