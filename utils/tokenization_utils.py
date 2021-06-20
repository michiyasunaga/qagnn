from transformers import PreTrainedTokenizer
import os
import nltk
import json
from tqdm import tqdm
import spacy

EOS_TOK = '<EOS>'
UNK_TOK = '<UNK>'
PAD_TOK = '<PAD>'
SEP_TOK = '<SEP>'
EXTRA_TOKS = [EOS_TOK, UNK_TOK, PAD_TOK, SEP_TOK]


class WordTokenizer(PreTrainedTokenizer):
    vocab_files_names = {'vocab_file': 'vocab.txt'}
    pretrained_vocab_files_map = {'vocab_file': {'lstm': './data/glove/glove.vocab'}}
    max_model_input_sizes = {'lstm': None}
    """
    vocab_file: Path to a json file that contains token-to-id mapping
    """

    def __init__(self, vocab_file, unk_token=UNK_TOK, sep_token=SEP_TOK, pad_token=PAD_TOK, eos_token=EOS_TOK, **kwargs):
        super(WordTokenizer, self).__init__(unk_token=unk_token, sep_token=sep_token,
                                            pad_token=pad_token, eos_token=eos_token, **kwargs)
        with open(vocab_file, 'r', encoding='utf-8') as fin:
            self.vocab = {line.rstrip('\n'): i for i, line in enumerate(fin)}
        self.ids_to_tokens = {ids: tok for tok, ids in self.vocab.items()}
        self.spacy_tokenizer = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        return tokenize_sentence_spacy(self.spacy_tokenizer, text, lower_case=True, convert_num=False)

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).strip()
        return out_string

    def add_special_tokens_single_sequence(self, token_ids):
        return token_ids + [self.eos_token_id]

    def add_special_tokens_sequence_pair(self, token_ids_0, token_ids_1):
        return token_ids_0 + [self.sep_token_id] + token_ids_1

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, self.vocab_files_names['vocab_file'])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as fout:
            for i in range(len(self.vocab)):
                fout.write(self.ids_to_tokens[i] + '\n')
        return (vocab_file,)


class WordVocab(object):

    def __init__(self, sents=None, path=None, freq_cutoff=5, encoding='utf-8', verbose=True):
        """
        sents: list[str] (optional, default None)
        path: str (optional, default None)
        freq_cutoff: int (optional, default 5, 0 to disable)
        encoding: str (optional, default utf-8)
        """
        if sents is not None:
            counts = {}
            for text in sents:
                for w in text.split():
                    counts[w] = counts.get(w, 0) + 1
            self._idx2w = [t[0] for t in sorted(counts.items(), key=lambda x: -x[1])]
            self._w2idx = {w: i for i, w in enumerate(self._idx2w)}
            self._counts = counts

        elif path is not None:
            self._idx2w = []
            self._counts = {}
            with open(path, 'r', encoding=encoding) as fin:
                for line in fin:
                    w, c = line.rstrip().split(' ')
                    self._idx2w.append(w)
                    self._counts[w] = c
                self._w2idx = {w: i for i, w in enumerate(self._idx2w)}

        else:
            self._idx2w = []
            self._w2idx = {}
            self._counts = {}

        if freq_cutoff > 1:
            self._idx2w = [w for w in self._idx2w if self._counts[w] >= freq_cutoff]

            in_sum = sum([self._counts[w] for w in self._idx2w])
            total_sum = sum([self._counts[w] for w in self._counts])
            if verbose:
                print('vocab oov rate: {:.4f}'.format(1 - in_sum / total_sum))

            self._w2idx = {w: i for i, w in enumerate(self._idx2w)}
            self._counts = {w: self._counts[w] for w in self._idx2w}

    def add_word(self, w, count=1):
        if w not in self.w2idx:
            self._w2idx[w] = len(self._idx2w)
            self._idx2w.append(w)
            self._counts[w] = count
        else:
            self._counts[w] += count
        return self

    def top_k_cutoff(self, size):
        if size < len(self._idx2w):
            for w in self._idx2w[size:]:
                self._w2idx.pop(w)
                self._counts.pop(w)
            self._idx2w = self._idx2w[:size]

        assert len(self._idx2w) == len(self._w2idx) == len(self._counts)
        return self

    def save(self, path, encoding='utf-8'):
        with open(path, 'w', encoding=encoding) as fout:
            for w in self._idx2w:
                fout.write(w + ' ' + str(self._counts[w]) + '\n')

    def __len__(self):
        return len(self._idx2w)

    def __contains__(self, word):
        return word in self._w2idx

    def __iter__(self):
        for word in self._idx2w:
            yield word

    @property
    def w2idx(self):
        return self._w2idx

    @property
    def idx2w(self):
        return self._idx2w

    @property
    def counts(self):
        return self._counts


def tokenize_sentence_nltk(sent, lower_case=True, convert_num=False):
    tokens = nltk.word_tokenize(sent)
    if lower_case:
        tokens = [t.lower() for t in tokens]
    if convert_num:
        tokens = ['<NUM>' if t.isdigit() else t for t in tokens]
    return tokens


def tokenize_sentence_spacy(nlp, sent, lower_case=True, convert_num=False):
    tokens = [tok.text for tok in nlp(sent)]
    if lower_case:
        tokens = [t.lower() for t in tokens]
    if convert_num:
        tokens = ['<NUM>' if t.isdigit() else t for t in tokens]
    return tokens


def tokenize_statement_file(statement_path, output_path, lower_case=True, convert_num=False):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    nrow = sum(1 for _ in open(statement_path, 'r'))
    with open(statement_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, total=nrow, desc='tokenizing'):
            data = json.loads(line)
            for statement in data['statements']:
                tokens = tokenize_sentence_spacy(nlp, statement['statement'], lower_case=lower_case, convert_num=convert_num)
                fout.write(' '.join(tokens) + '\n')


def make_word_vocab(statement_path_list, output_path, lower_case=True, convert_num=True, freq_cutoff=5):
    """save the vocab to the output_path in json format"""
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])

    docs = []
    for path in statement_path_list:
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                json_dic = json.loads(line)
                docs += [json_dic['question']['stem']] + [s['text'] for s in json_dic['question']['choices']]

    counts = {}
    for doc in tqdm(docs, desc='making word vocab'):
        for w in tokenize_sentence_spacy(nlp, doc, lower_case=lower_case, convert_num=convert_num):
            counts[w] = counts.get(w, 0) + 1
    idx2w = [t[0] for t in sorted(counts.items(), key=lambda x: -x[1])]
    idx2w = [w for w in idx2w if counts[w] >= freq_cutoff]
    idx2w += EXTRA_TOKS
    w2idx = {w: i for i, w in enumerate(idx2w)}
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(w2idx, fout)


def run_test():
    # tokenize_statement_file('data/csqa/statement/dev.statement.jsonl', '/tmp/tokenized.txt', True, True)
    # make_word_vocab(['data/csqa/statement/dev.statement.jsonl', 'data/csqa/statement/train.statement.jsonl'], '/tmp/vocab.txt', True, True)
    tokenizer = WordTokenizer.from_pretrained('lstm')
    print(tokenizer.tokenize('I love NLP since 1998DEC'))
    print(tokenizer.tokenize('CXY loves NLP since 1998'))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('CXY loves NLP since 1998')))
    print(tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('CXY loves NLP since 1998'))))
    tokenizer.save_pretrained('/tmp/')
    tokenizer = WordTokenizer.from_pretrained('/tmp/')
    print('vocab size = {}'.format(tokenizer.vocab_size))


if __name__ == '__main__':
    run_test()
