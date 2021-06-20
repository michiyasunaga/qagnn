from multiprocessing import Pool
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json
import string


__all__ = ['create_matcher_patterns', 'ground']


# the lemma of it/them/mine/.. is -PRON-

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

# CHUNK_SIZE = 1

CPNET_VOCAB = None
PATTERN_PATH = None
nlp = None
matcher = None


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


def create_pattern(nlp, doc, debug=False):
    pronoun_list = set(["my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our", "we"])
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or \
            all([(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords or token.lemma_ in blacklist) for token in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


def create_matcher_patterns(cpnet_vocab_path, output_path, debug=False):
    cpnet_vocab = load_cpnet_vocab(cpnet_vocab_path)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(cpnet_vocab)
    all_patterns = {}

    if debug:
        f = open("filtered_concept.txt", "w")

    for doc in tqdm(docs, total=len(cpnet_vocab)):

        pattern = create_pattern(nlp, doc, debug)
        if debug:
            if not pattern[0]:
                f.write(pattern[1] + '\n')

        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern

    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    if debug:
        f.close()


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    # for i in range(len(doc)):
    #     lemmas = []
    #     for j, token in enumerate(doc):
    #         if j == i:
    #             lemmas.append(token.lemma_)
    #         else:
    #             lemmas.append(token.text)
    #     lc = "_".join(lemmas)
    #     lcs.add(lc)
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp, pattern_path):
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, None, pattern)
    return matcher


def ground_qa_pair(qa_pair):
    global nlp, matcher
    if nlp is None or matcher is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        matcher = load_matcher(nlp, PATTERN_PATH)

    s, a = qa_pair
    all_concepts = ground_mentioned_concepts(nlp, matcher, s, a)
    answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
    question_concepts = all_concepts - answer_concepts
    if len(question_concepts) == 0:
        question_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible

    if len(answer_concepts) == 0:
        answer_concepts = hard_ground(nlp, a, CPNET_VOCAB)  # some case

    # question_concepts = question_concepts -  answer_concepts
    question_concepts = sorted(list(question_concepts))
    answer_concepts = sorted(list(answer_concepts))
    return {"sent": s, "ans": a, "qc": question_concepts, "ac": answer_concepts}


def ground_mentioned_concepts(nlp, matcher, s, ans=None):

    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    if ans is not None:
        ans_matcher = Matcher(nlp.vocab)
        ans_words = nlp(ans)
        # print(ans_words)
        ans_matcher.add(ans, None, [{'TEXT': token.text.lower()} for token in ans_words])

        ans_match = ans_matcher(doc)
        ans_mentions = set()
        for _, ans_start, ans_end in ans_match:
            ans_mentions.add((ans_start, ans_end))

    for match_id, start, end in matches:
        if ans is not None:
            if (start, end) in ans_mentions:
                continue

        span = doc[start:end].text  # the matched span

        # a word that appears in answer is not considered as a mention in the question
        # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
        #     continue
        original_concept = nlp.vocab.strings[match_id]
        original_concept_set = set()
        original_concept_set.add(original_concept)

        # print("span", span)
        # print("concept", original_concept)
        # print("Matched '" + span + "' to the rule '" + string_id)

        # why do you lemmatize a mention whose len == 1?

        if len(original_concept.split("_")) == 1:
            # tag = doc[start].tag_
            # if tag in ['VBN', 'VBG']:

            original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].update(original_concept_set)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        # print("span:")
        # print(span)
        # print("concept_sorted:")
        # print(concepts_sorted)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3]

        for c in shortest:
            if c in blacklist:
                continue

            # a set with one string like: set("like_apples")
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)

        # if a mention exactly matches with a concept

        exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
        # print("exact match:")
        # print(exact_match)
        assert len(exact_match) < 2
        mentioned_concepts.update(exact_match)

    return mentioned_concepts


def hard_ground(nlp, sent, cpnet_vocab):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    try:
        assert len(res) > 0
    except Exception:
        print(f"for {sent}, concept not found in hard grounding.")
    return res


def match_mentioned_concepts(sents, answers, num_processes):
    res = []
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(ground_qa_pair, zip(sents, answers)), total=len(sents)))
    return res


# To-do: examine prune
def prune(data, cpnet_vocab_path):
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    prune_data = []
    for item in tqdm(data):
        qc = item["qc"]
        prune_qc = []
        for c in qc:
            if c[-2:] == "er" and c[:-2] in qc:
                continue
            if c[-1:] == "e" and c[:-1] in qc:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords:
                    have_stop = True
            if not have_stop and c in cpnet_vocab:
                prune_qc.append(c)

        ac = item["ac"]
        prune_ac = []
        for c in ac:
            if c[-2:] == "er" and c[:-2] in ac:
                continue
            if c[-1:] == "e" and c[:-1] in ac:
                continue
            all_stop = True
            for t in c.split("_"):
                if t not in nltk_stopwords:
                    all_stop = False
            if not all_stop and c in cpnet_vocab:
                prune_ac.append(c)

        try:
            assert len(prune_ac) > 0 and len(prune_qc) > 0
        except Exception as e:
            pass
            # print("In pruning")
            # print(prune_qc)
            # print(prune_ac)
            # print("original:")
            # print(qc)
            # print(ac)
            # print()
        item["qc"] = prune_qc
        item["ac"] = prune_ac

        prune_data.append(item)
    return prune_data


def ground(statement_path, cpnet_vocab_path, pattern_path, output_path, num_processes=1, debug=False):
    global PATTERN_PATH, CPNET_VOCAB
    if PATTERN_PATH is None:
        PATTERN_PATH = pattern_path
        CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)

    sents = []
    answers = []
    with open(statement_path, 'r') as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[192:195]
        print(len(lines))
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        # {'answerKey': 'B',
        #   'id': 'b8c0a4703079cf661d7261a60a1bcbff',
        #   'question': {'question_concept': 'magazines',
        #                 'choices': [{'label': 'A', 'text': 'doctor'}, {'label': 'B', 'text': 'bookstore'}, {'label': 'C', 'text': 'market'}, {'label': 'D', 'text': 'train station'}, {'label': 'E', 'text': 'mortuary'}],
        #                 'stem': 'Where would you find magazines along side many other printed works?'},
        #   'statements': [{'label': False, 'statement': 'Doctor would you find magazines along side many other printed works.'}, {'label': True, 'statement': 'Bookstore would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Market would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Train station would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Mortuary would you find magazines along side many other printed works.'}]}

        for statement in j["statements"]:
            sents.append(statement["statement"])

        for answer in j["question"]["choices"]:
            ans = answer['text']
            # ans = " ".join(answer['text'].split("_"))
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            answers.append(ans)

    res = match_mentioned_concepts(sents, answers, num_processes)
    res = prune(res, cpnet_vocab_path)

    # check_path(output_path)
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == "__main__":
    create_matcher_patterns("../data/cpnet/concept.txt", "./matcher_res.txt", True)
    # ground("../data/statement/dev.statement.jsonl", "../data/cpnet/concept.txt", "../data/cpnet/matcher_patterns.json", "./ground_res.jsonl", 10, True)

    # s = "a revolving door is convenient for two direction travel, but it also serves as a security measure at a bank."
    # a = "bank"
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # ans_words = nlp(a)
    # doc = nlp(s)
    # ans_matcher = Matcher(nlp.vocab)
    # print([{'TEXT': token.text.lower()} for token in ans_words])
    # ans_matcher.add("ok", None, [{'TEXT': token.text.lower()} for token in ans_words])
    #
    # matches = ans_matcher(doc)
    # for a, b, c in matches:
    #     print(a, b, c)
