import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from utils.utils import freeze_net


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class TypedLinear(nn.Linear):
    def __init__(self, in_features, out_features, n_type):
        super().__init__(in_features, n_type * out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.n_type = n_type

    def forward(self, X, type_ids=None):
        """
        X: tensor of shape (*, in_features)
        type_ids: long tensor of shape (*)
        """
        output = super().forward(X)
        if type_ids is None:
            return output
        output_shape = output.size()[:-1] + (self.out_features,)
        output = output.view(-1, self.n_type, self.out_features)
        idx = torch.arange(output.size(0), dtype=torch.long, device=type_ids.device)
        output = output[idx, type_ids.view(-1)].view(*output_shape)
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled


class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size=300, emb_size=300, hidden_size=300, num_layers=2, bidirectional=True,
                 emb_p=0, input_p=0, hidden_p=0, output_p=0, pretrained_emb=None, pooling=True, pad=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.emb_p = emb_p
        self.input_p = input_p
        self.hidden_p = hidden_p
        self.output_p = output_p
        self.pooling = pooling

        self.emb = EmbeddingDropout(nn.Embedding(vocab_size, emb_size), emb_p)
        if pretrained_emb is not None:
            self.emb.emb.weight.data.copy_(pretrained_emb)
        else:
            bias = np.sqrt(6.0 / emb_size)
            nn.init.uniform_(self.emb.emb.weight, -bias, bias)
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=(hidden_size // 2 if self.bidirectional else hidden_size),
                           num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                           batch_first=True)
        self.max_pool = MaxPoolLayer()

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bz, full_length = inputs.size()
        embed = self.emb(inputs)
        embed = self.input_dropout(embed)
        lstm_inputs = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.rnn(lstm_inputs)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True, total_length=full_length)
        rnn_outputs = self.output_dropout(rnn_outputs)
        return self.max_pool(rnn_outputs, lengths) if self.pooling else rnn_outputs


class TripleEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, input_p, output_p, hidden_p, num_layers, bidirectional=True, pad=False,
                 concept_emb=None, relation_emb=None
                 ):
        super().__init__()
        if pad:
            raise NotImplementedError
        self.input_p = input_p
        self.output_p = output_p
        self.hidden_p = hidden_p
        self.cpt_emb = concept_emb
        self.rel_emb = relation_emb
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=(hidden_dim // 2 if self.bidirectional else hidden_dim),
                          num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                          batch_first=True)

    def forward(self, inputs):
        '''
        inputs: (batch_size, seq_len)

        returns: (batch_size, h_dim(*2))
        '''
        bz, sl = inputs.size()
        h, r, t = torch.chunk(inputs, 3, dim=1)  # (bz, 1)

        h, t = self.input_dropout(self.cpt_emb(h)), self.input_dropout(self.cpt_emb(t))  # (bz, 1, dim)
        r = self.input_dropout(self.rel_emb(r))
        inputs = torch.cat((h, r, t), dim=1)  # (bz, 3, dim)
        rnn_outputs, _ = self.rnn(inputs)  # (bz, 3, dim)
        if self.bidirectional:
            outputs_f, outputs_b = torch.chunk(rnn_outputs, 2, dim=2)
            outputs = torch.cat((outputs_f[:, -1, :], outputs_b[:, 0, :]), 1)  # (bz, 2 * h_dim)
        else:
            outputs = rnn_outputs[:, -1, :]

        return self.output_dropout(outputs)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class AttPoolLayer(nn.Module):

    def __init__(self, d_q, d_k, dropout=0.1):
        super().__init__()
        self.w_qs = nn.Linear(d_q, d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_k)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q)
        k: tensor of shape (b, l, d_k)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, d_k)
        """
        qs = self.w_qs(q)  # (b, d_k)
        output, attn = self.attention(qs, k, k, mask=mask)
        output = self.dropout(output)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class TypedMultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1, n_type=1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = TypedLinear(d_k_original, n_head * self.d_k, n_type)
        self.w_vs = TypedLinear(d_k_original, n_head * self.d_v, n_type)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None, type_ids=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: bool tensor of shape (b, l) (optional, default None)
        type_ids: long tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k, type_ids=type_ids).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k, type_ids=type_ids).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class BilinearAttentionLayer(nn.Module):

    def __init__(self, query_dim, value_dim):
        super().__init__()
        self.linear = nn.Linear(value_dim, query_dim, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, query, value, node_mask=None):
        """
        query: tensor of shape (batch_size, query_dim)
        value: tensor of shape (batch_size, seq_len, value_dim)
        node_mask: tensor of shape (batch_size, seq_len)

        returns: tensor of shape (batch_size, value_dim)
        """
        attn = self.linear(value).bmm(query.unsqueeze(-1))
        attn = self.softmax(attn.squeeze(-1))
        if node_mask is not None:
            attn = attn * node_mask
            attn = attn / attn.sum(1, keepdim=True)
        pooled = attn.unsqueeze(1).bmm(value).squeeze(1)
        return pooled, attn


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = True,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # # To limit numerical errors from large vector elements outside the mask, we zero these out.
            # result = nn.functional.softmax(vector * mask, dim=dim)
            # result = result * mask
            # result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            raise NotImplementedError
        else:
            masked_vector = vector.masked_fill(mask.to(dtype=torch.uint8), mask_fill_value)
            result = nn.functional.softmax(masked_vector, dim=dim)
            result = result * (1 - mask)
    return result


class DiffTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        """
        x: tensor of shape (batch_size, n_node)
        k: int
        returns: tensor of shape (batch_size, n_node)
        """
        bs, _ = x.size()
        _, topk_indexes = x.topk(k, 1)  # (batch_size, k)
        output = x.new_zeros(x.size())
        ri = torch.arange(bs).unsqueeze(1).expand(bs, k).contiguous().view(-1)
        output[ri, topk_indexes.view(-1)] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class SimilarityFunction(nn.Module):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.
    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.
    """
    default_implementation = 'dot_product'

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError


class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """

    def __init__(self, scale_output: bool = False) -> None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result


class MatrixAttention(nn.Module):
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        self._similarity_function = similarity_function or DotProductSimilarity()

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)


class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


def run_test():
    print('testing BilinearAttentionLayer...')
    att = BilinearAttentionLayer(100, 20)
    mask = (torch.randn(70, 30) > 0).float()
    mask.requires_grad_()
    v = torch.randn(70, 30, 20)
    q = torch.randn(70, 100)
    o, _ = att(q, v, mask)
    o.sum().backward()
    print(mask.grad)

    print('testing DiffTopK...')
    x = torch.randn(5, 3)
    x.requires_grad_()
    k = 2
    r = DiffTopK.apply(x, k)
    loss = (r ** 2).sum()
    loss.backward()
    assert (x.grad == r * 2).all()
    print('pass')

    a = TripleEncoder()

    triple_input = torch.tensor([[1, 2, 3], [4, 5, 6]])
    res = a(triple_input)
    print(res.size())

    b = LSTMEncoder(pooling=False)
    lstm_inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    lengths = torch.tensor([3, 2])
    res = b(lstm_inputs, lengths)
    print(res.size())
