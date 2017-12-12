import torch
import pickle
import sys
import numpy as np

from data import *

word_embedding = torch.load("word_embedding.p")
#model = torch.load("model.p")

with open("test_queries.p", "rb") as f:
    test_queries = pickle.load(f)
with open("test_labels.p", "rb") as f:
    test_labels = pickle.load(f)
with open("corpus.p", "rb") as f:
    corpus = pickle.load(f)
with open("title2idx.p", "rb") as f:
    title2idx = pickle.load(f)
    idx2title = {v:k for k,v in title2idx.items()}
    del title2idx
with open("vocab.p", "rb") as f:
    vocab = pickle.load(f)

def filter_empty():
    for i in range(len(test_queries)-1, -1, -1):
        if len(test_queries[i]) == 0 or len(test_labels[i]) == 0:
            del test_queries[i], test_labels[i]

filter_empty()

print("test dataset = %d" % len(test_queries))
#test_dataset = MSADataset(test_queries, test_labels, corpus)
#print("test dataset = {}".format(len(test_dataset)))

def score(W, x):
    return W @ x


def avg_embedding(query):
    if len(query) == 0:
        return np.zeros(word_embedding.shape[1])
    embeddings = word_embedding[1:][np.array(query)]
    return embeddings.mean(0)


def compute_corpus_avg_embedding():
    n_corpus = len(corpus)
    dim = word_embedding.shape[1]
    corpus_embedding = []
    for i in range(n_corpus):
        avg = avg_embedding(corpus[i])
        corpus_embedding.append(avg)
    return np.array(corpus_embedding)

corpus_embedding = compute_corpus_avg_embedding()


def precision_recall(ground_truth, predicted):
    n = len(ground_truth)
    k = len(predicted)
    precision = []
    recall = []
    rel = [0] * k
    
    p = 0
    r = 0
    hit = 0
    for i in range(k):
        if predicted[i] in ground_truth:
            rel[i] = 1
            hit += 1
            r += 1. / n
        p = hit / (i+1.)

        precision.append(p)
        recall.append(r)

    return precision, recall, rel


def avg_precision(precision, rel):
    if sum(rel) == 0:
        return 0

    avg_p = sum([precision[i] * rel[i] for i in range(len(rel))]) / sum(rel)
    return avg_p


def MAP(avg_p):
    return sum(avg_p) / len(avg_p)


def evaluate(k=10):
    np.random.seed(111111)
    indices = np.random.choice(len(test_queries), 100)
    indices = {i:1 for i in indices}
    total_avg_p = []

    for i in range(len(test_queries)):
        q = avg_embedding(test_queries[i])
        s = score(corpus_embedding, q)
        top_k = np.argsort(s)[-k:]

        precision, recall, rel = precision_recall(test_labels[i], top_k)
        n = len(test_labels[i])
        avg_p = avg_precision(precision, rel)
        total_avg_p.append(avg_p)

        if i in indices:
            print("query")
            query = " ".join([vocab[idx] for idx in test_queries[i]])
            print(query)
            print("ground truth")
            print([idx2title[idx] for idx in test_labels[i]])
            print("output")
            print([idx2title[idx] for idx in top_k])

    mean_ap = MAP(total_avg_p)
    print("mean avg precision: {:2.6f}".format(mean_ap))

        

if __name__ == "__main__":
    evaluate()
