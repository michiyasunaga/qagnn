import networkx as nx
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import sys

try:
    from .utils import check_file
except ImportError:
    from utils import check_file

__all__ = ['extract_english', 'construct_graph', 'merged_relations']

relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
]


def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_english(conceptnet_path, output_csv_path, output_vocab_path):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print('extracting English concepts and relations from ConceptNet...')
    relation_mapping = load_merge_relation()
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    cpnet_vocab = []
    concepts_seen = set()
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_csv_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin, total=num_lines):
            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = toks[1].split("/")[-1].lower()
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if rel not in relation_mapping:
                    continue

                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    head, tail, rel = tail, head, rel[1:]

                data = json.loads(toks[4])

                fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')

                for w in [head, tail]:
                    if w not in concepts_seen:
                        concepts_seen.add(w)
                        cpnet_vocab.append(w)

    with open(output_vocab_path, 'w') as fout:
        for word in cpnet_vocab:
            fout.write(word + '\n')

    print(f'extracted ConceptNet csv file saved to {output_csv_path}')
    print(f'extracted concept vocabulary saved to {output_vocab_path}')
    print()


def construct_graph(cpnet_csv_path, cpnet_vocab_path, output_path, prune=True):
    print('generating ConceptNet graph file...')

    nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                       "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py

    blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py

    concept2id = {}
    id2concept = {}
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in blacklist:
                return True
            '''originally phrases like "branch out" would not be kept in the graph'''
            # for t in cpt.split("_"):
            #     if t in nltk_stopwords:
            #         return True
            return False

        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
            # weight -= 0.3
            # continue
            if subj == obj:  # delete loops
                continue
            # weight = 1 + float(math.exp(1 - weight))  # issue: ???

            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                attrs.add((obj, subj, rel + len(relation2id)))

    nx.write_gpickle(graph, output_path)
    print(f"graph file saved to {output_path}")
    print()


def glove_init(input, output, concept_file):
    embeddings_file = output + '.npy'
    vocabulary_file = output.split('.')[0] + '.vocab.txt'
    output_dir = '/'.join(output.split('/')[:-1])
    output_prefix = output.split('/')[-1]

    words = []
    vectors = []
    vocab_exist = check_file(vocabulary_file)
    print("loading embedding")
    with open(input, 'rb') as f:
        for line in f:
            fields = line.split()
            if len(fields) <= 2:
                continue
            if not vocab_exist:
                word = fields[0].decode('utf-8')
                words.append(word)
            vector = np.fromiter((float(x) for x in fields[1:]),
                                 dtype=np.float)

            vectors.append(vector)
        dim = vector.shape[0]
    print("converting")
    matrix = np.array(vectors, dtype="float32")
    print("writing")
    np.save(embeddings_file, matrix)
    text = '\n'.join(words)
    if not vocab_exist:
        with open(vocabulary_file, 'wb') as f:
            f.write(text.encode('utf-8'))

    def load_glove_from_npy(glove_vec_path, glove_vocab_path):
        vectors = np.load(glove_vec_path)
        with open(glove_vocab_path, "r", encoding="utf8") as f:
            vocab = [l.strip() for l in f.readlines()]

        assert (len(vectors) == len(vocab))

        glove_embeddings = {}
        for i in range(0, len(vectors)):
            glove_embeddings[vocab[i]] = vectors[i]
        print("Read " + str(len(glove_embeddings)) + " glove vectors.")
        return glove_embeddings

    def weighted_average(avg, new, n):
        # TODO: maybe a better name for this function?
        return ((n - 1) / n) * avg + (new / n)

    def max_pooling(old, new):
        # TODO: maybe a better name for this function?
        return np.maximum(old, new)

    def write_embeddings_npy(embeddings, embeddings_cnt, npy_path, vocab_path):
        words = []
        vectors = []
        for key, vec in embeddings.items():
            words.append(key)
            vectors.append(vec)

        matrix = np.array(vectors, dtype="float32")
        print(matrix.shape)

        print("Writing embeddings matrix to " + npy_path, flush=True)
        np.save(npy_path, matrix)
        print("Finished writing embeddings matrix to " + npy_path, flush=True)

        if not check_file(vocab_path):
            print("Writing vocab file to " + vocab_path, flush=True)
            to_write = ["\t".join([w, str(embeddings_cnt[w])]) for w in words]
            with open(vocab_path, "w", encoding="utf8") as f:
                f.write("\n".join(to_write))
            print("Finished writing vocab file to " + vocab_path, flush=True)

    def create_embeddings_glove(pooling="max", dim=100):
        print("Pooling: " + pooling)

        with open(concept_file, "r", encoding="utf8") as f:
            triple_str_json = json.load(f)
        print("Loaded " + str(len(triple_str_json)) + " triple strings.")

        glove_embeddings = load_glove_from_npy(embeddings_file, vocabulary_file)
        print("Loaded glove.", flush=True)

        concept_embeddings = {}
        concept_embeddings_cnt = {}
        rel_embeddings = {}
        rel_embeddings_cnt = {}

        for i in tqdm(range(len(triple_str_json))):
            data = triple_str_json[i]

            words = data["string"].strip().split(" ")

            rel = data["rel"]
            subj_start = data["subj_start"]
            subj_end = data["subj_end"]
            obj_start = data["obj_start"]
            obj_end = data["obj_end"]

            subj_words = words[subj_start:subj_end]
            obj_words = words[obj_start:obj_end]

            subj = " ".join(subj_words)
            obj = " ".join(obj_words)

            # counting the frequency (only used for the avg pooling)
            if subj not in concept_embeddings:
                concept_embeddings[subj] = np.zeros((dim,))
                concept_embeddings_cnt[subj] = 0
            concept_embeddings_cnt[subj] += 1

            if obj not in concept_embeddings:
                concept_embeddings[obj] = np.zeros((dim,))
                concept_embeddings_cnt[obj] = 0
            concept_embeddings_cnt[obj] += 1

            if rel not in rel_embeddings:
                rel_embeddings[rel] = np.zeros((dim,))
                rel_embeddings_cnt[rel] = 0
            rel_embeddings_cnt[rel] += 1

            if pooling == "avg":
                subj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in subj])
                obj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in obj])

                if rel in ["relatedto", "antonym"]:
                    # Symmetric relation.
                    rel_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in
                                            words]) - subj_encoding_sum - obj_encoding_sum
                else:
                    # Asymmetrical relation.
                    rel_encoding_sum = obj_encoding_sum - subj_encoding_sum

                subj_len = subj_end - subj_start
                obj_len = obj_end - obj_start

                subj_encoding = subj_encoding_sum / subj_len
                obj_encoding = obj_encoding_sum / obj_len
                rel_encoding = rel_encoding_sum / (len(words) - subj_len - obj_len)

                concept_embeddings[subj] = subj_encoding
                concept_embeddings[obj] = obj_encoding
                rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

            elif pooling == "max":
                subj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in subj_words], axis=0)
                obj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in obj_words], axis=0)

                mask_rel = []
                for j in range(len(words)):
                    if subj_start <= j < subj_end or obj_start <= j < obj_end:
                        continue
                    mask_rel.append(j)
                rel_vecs = [glove_embeddings.get(words[i], np.zeros((dim,))) for i in mask_rel]
                rel_encoding = np.amax(rel_vecs, axis=0)

                # here it is actually avg over max for relation
                concept_embeddings[subj] = max_pooling(concept_embeddings[subj], subj_encoding)
                concept_embeddings[obj] = max_pooling(concept_embeddings[obj], obj_encoding)
                rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

        print(str(len(concept_embeddings)) + " concept embeddings")
        print(str(len(rel_embeddings)) + " relation embeddings")

        write_embeddings_npy(concept_embeddings, concept_embeddings_cnt, f'{output_dir}/concept.{output_prefix}.{pooling}.npy',
                             f'{output_dir}/concept.glove.{pooling}.txt')
        write_embeddings_npy(rel_embeddings, rel_embeddings_cnt, f'{output_dir}/relation.{output_prefix}.{pooling}.npy',
                             f'{output_dir}/relation.glove.{pooling}.txt')

    create_embeddings_glove(dim=dim)


if __name__ == "__main__":
    glove_init("../data/glove/glove.6B.200d.txt", "../data/glove/glove.200d", '../data/glove/tp_str_corpus.json')
