from functools import partial
from torch.utils.data import DataLoader
import pickle
import copy

from bow import BOW
from data import *

##########
# config
##########
batch_size = 128
embed_size = 50
epochs = 10
l2_lambda = 1e-5
log_interval = 100


##########
# load vocabulary & data
##########
with open('corpus.p', 'rb') as f:
    corpus = pickle.load(f)

dset = ['train', 'valid']
dataset = {k: [] for k in dset}
for d in dset:
    with open(d+'_queries.p', 'rb') as f:
        dataset[d].append(pickle.load(f))
    with open(d+'_labels.p', 'rb') as f:
        dataset[d].append(pickle.load(f))

with open('vocab.p', 'rb') as f:
    vocab = pickle.load(f)
    print("vocab size = {}".format(len(vocab)))


##########
# define dataset & dataloader
##########
train_dataset = MSADataset(dataset['train'][0], dataset['train'][1], corpus)
valid_dataset = MSADataset(dataset['valid'][0], dataset['valid'][1], corpus)
print("training data = {}".format(len(train_dataset)))
print("validation data = {}".format(len(valid_dataset)))

MSADataLoader = partial(DataLoader, collate_fn=collate2)
train_dataloader = MSADataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = MSADataLoader(valid_dataset, batch_size=batch_size)


##########
# define model
##########
model = BOW(len(vocab), embed_size, len(corpus))
model = torch.load("model3.p")
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

def hasnan(x):
    n = (x.data != x.data).sum()
    return n != 0


def clip_grad(grad, clip=20):
    thres = torch.ones(grad.data.size())*clip
    if cuda:
        thres = thres.cuda()
    grad.data = torch.min(grad.data, thres)
    #grad.data.clamp_(-clip, clip)


best_valid_loss = np.inf
best_model = None

for epoch in range(epochs):
    ## training
    model.train()

    train_loss = 0
    batch = 1
    for q, q_len, text_pos, pos_len, text_neg, neg_len in train_dataloader:
        if cuda:
            q = q.cuda()
            q_len = q_len.cuda()
            text_pos = text_pos.cuda()
            pos_len = pos_len.cuda()
            text_neg = text_neg.cuda()
            neg_len = neg_len.cuda()

        loss, w_norms = model.loss(q, q_len, text_pos, pos_len, text_neg, neg_len)
        total_loss = loss + l2_lambda * w_norms
        assert not hasnan(total_loss)
        
        opt.zero_grad()
        total_loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                clip_grad(p.grad)
                assert not hasnan(p.grad)
        opt.step()

        train_loss += np.asscalar(loss.data.cpu().numpy()) * len(q)

        if batch % log_interval == 0:
            batch_data_size = len(q) * batch
            print("Batches [{:5d}/{:5d}], loss = {:5.6f}".format(
                batch_data_size, len(train_dataset), train_loss/batch_data_size))
        batch += 1

    train_loss /= len(train_dataset)

    ## validation
    model.eval()

    valid_loss = 0
    for q, q_len, text_pos, pos_len, text_neg, neg_len in valid_dataloader:
        if cuda:
            q = q.cuda()
            q_len = q_len.cuda()
            text_pos = text_pos.cuda()
            pos_len = pos_len.cuda()
            text_neg = text_neg.cuda()
            neg_len = neg_len.cuda()

        loss, _ = model.loss(q, q_len, text_pos, pos_len, text_neg, neg_len)
        valid_loss += np.asscalar(loss.data.cpu().numpy()) * len(q)
    valid_loss /= len(valid_dataset)

    print("[Epoch {:5d}] Training loss = {:5.6f}\tValidation loss = {:5.6f}".format(
        epoch, train_loss, valid_loss))

    ## saving
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model = copy.deepcopy(model)
        torch.save(best_model.word_embedding.weight.data.cpu().numpy(), "word_embedding.p")
        torch.save(best_model.cpu(), "model3_cpu.p")
        torch.save(best_model, "model3.p")
