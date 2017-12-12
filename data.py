import h5py
import re
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class MSADataset(Dataset):

    def __init__(self, query, label, corpus, k=20):
        assert len(query) == len(label)

        self.query = query
        self.label = label
        self.corpus = corpus
        self.k = k


    def __len__(self):
        return len(self.query)


    def __getitem__(self, i):
        def _flatten(lists):
            return [x for l in lists for x in l]

        q = self.query[i]
        text_pos = [self.corpus[j] for j in self.label[i]]
        neg_label = np.random.choice(len(self), self.k)
        text_neg = [self.corpus[j] for j in neg_label]
        return q, _flatten(text_pos), _flatten(text_neg)


def collate2(samples):
    def _pad(x):
        seq_lens = [len(i) for i in x]
        max_len = max(seq_lens)

        for i in range(len(x)):
            x[i] = np.pad(
                    x[i], 
                    (0, max_len - len(x[i])), 
                    'constant', 
                    constant_values=-1
                    )

        return np.array(x, dtype=int)+1, np.array(seq_lens, dtype=int)

    query, text_pos, text_neg = [list(x) for x in zip(*samples)]
    query, q_len = _pad(query)
    text_pos, text_pos_len = _pad(text_pos)
    text_neg, text_neg_len = _pad(text_neg)

    return (Variable(torch.LongTensor(query)), 
            Variable(torch.LongTensor(q_len)), 
            Variable(torch.LongTensor(text_pos)), 
            Variable(torch.LongTensor(text_pos_len)), 
            Variable(torch.LongTensor(text_neg)), 
            Variable(torch.LongTensor(text_neg_len))
            )


def collate(samples):
    def flatten(lists):
        flat_list = []
        for l in lists:
            flat_list.append([v for x in l for v in x])
        return flat_list

    def _pad(x):
        '''
        x: (batch_size, num, seq_len)
        '''
        batch_size = len(x)
        max_n = max([len(i) for i in x])
        seq_lens = -np.ones([batch_size, max_n], dtype=int)
        max_len = 0
        for batch in range(batch_size):
            seq_lens[batch][:len(x[batch])] = np.array([len(i) for i in x[batch]])
            max_len = max(max_len, int(max(seq_lens[batch])))

        padded = np.zeros([batch_size, max_n, max_len], dtype=int) - 1

        for batch in range(batch_size):
            for i in range(len(x[batch])):
                padded[batch][i][:len(x[batch][i])] = np.array(x[batch][i])
                '''np.pad(
                        x[batch][i], 
                        (0, max_len - len(x[batch][i])), 
                        'constant', 
                        constant_values=-1
                        )
                '''
        return seq_lens, padded+1

    query, text_pos, text_neg = [list(x) for x in zip(*samples)]
    query = [[q] for q in query]
    q_len, query = _pad(query)
    text_pos_len, text_pos = _pad(text_pos)
    text_neg_len, text_neg = _pad(text_neg)

    return (Variable(torch.LongTensor(query)).squeeze(), 
            Variable(torch.LongTensor(q_len)), 
            Variable(torch.LongTensor(text_pos)), 
            Variable(torch.LongTensor(text_pos_len)), 
            Variable(torch.LongTensor(text_neg)), 
            Variable(torch.LongTensor(text_neg_len))
            )


def load_dataset(path):
    f = h5py.File(path, 'r')

    dset = ['train', 'valid', 'test']
    queries = []
    doc_ids = []
    for d in dset:
        queries.append(list(f['queries_'+d]))
        doc_ids.append(list(f['doc_ids_'+d]))

    return queries, doc_ids


def load_corpus(path):
    f = h5py.File(path, 'r')

    text = f['text']
    title = f['title']
    return title, text


def mapping(title):
    title2idx = dict((t, i) for i, t, in enumerate(title.value))
    idx2title = {v: k for k, v in title2idx.items()}
    return title2idx, idx2title


def lowercase(corpus):
    return [c.decode('utf-8').lower() for c in corpus]


def tokenize(corpus):
    from nltk.tokenize import word_tokenize
    return [word_tokenize(c) for c in corpus]


def filter_stopword(corpus):
    from nltk.corpus import stopwords
    sw_list = stopwords.words('english') + ["n't", "'ll", "'s", "'d"]
    corpus = [list(filter(lambda x: x not in sw_list and len(x) > 1, c)) for c in corpus]
    return corpus


def stem(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    stemmed = []
    for c in corpus:
        stemmed.append([stemmer.stem(w) for w in c])
    return stemmed


def build_vocab(corpus):
    from collections import Counter
    vocab = Counter()
    for sent in corpus:
        vocab.update(sent)
    return vocab


def preprocess(corpus, stopword=True, stemming=True):
    corpus = lowercase(corpus)
    corpus = tokenize(corpus)
    if stopword:
        corpus = filter_stopword(corpus)
    if stemming:
        corpus = stem(corpus)
    vocab = build_vocab(corpus)

    return corpus, vocab


def listify(doc_ids, title2idx):
    regex = re.compile(b"u'(.*?)'", re.S)
    doc_titles = re.findall(regex, doc_ids)
    title_ids = [title2idx[dt] for dt in doc_titles if dt in title2idx]
    return title_ids


def digitify_corpus(corpus, vocab):
    digit_corpus = []
    for s in corpus:
        c = [vocab[w] for w in s]
        digit_corpus.append(c)
    return digit_corpus
        

if __name__ == '__main__':
    import pickle

    title, text = load_corpus('msa_corpus.hdf5')
    title2idx, idx2title = mapping(title)
    #title2idx = pickle.load(open('title2idx.p', 'rb'))
    print('corpus size = {}'.format(len(title2idx)))
    pickle.dump(title2idx, open('title2idx.p', 'wb'))

    queries, doc_ids = load_dataset('msa_dataset.hdf5')
    corpus, vocab = preprocess(text)
    vocab.update(preprocess(title)[1])
    vocab = dict(enumerate(vocab))
    #vocab = pickle.load(open('vocab.p', 'rb'))
    print('vocab size = {}'.format(len(vocab)))
    pickle.dump(vocab, open('vocab.p', 'wb'))

    word2idx = {v: k for k, v in vocab.items()}
    #corpus = pickle.load(open('corpus.p', 'rb'))
    corpus = digitify_corpus(corpus, word2idx)
    print('digitified corpus {}'.format(len(corpus)))
    pickle.dump(corpus, open('corpus.p', 'wb'))

    prefix = ['train', 'valid', 'test']
    for i in range(len(queries)):
        outputs = []
        q, _ = preprocess(queries[i])
        query = digitify_corpus(q, word2idx)
        print(len(query))
        for j in range(len(queries[i])):
            label = listify(doc_ids[i][j], title2idx)
            outputs.append(label)

        print('{} dataset generated'.format(prefix[i]))
        pickle.dump(query, open(prefix[i]+'_queries.p', 'wb'))
        pickle.dump(outputs, open(prefix[i]+'_labels.p', 'wb'))
