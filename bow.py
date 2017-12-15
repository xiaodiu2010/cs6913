import torch
import torch.nn as nn

  

  
class BOW(nn.Module):

    def __init__(self, vocab_size, embed_size, n_corpus):
        nn.Module.__init__(self)
        self.n_corpus = n_corpus
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size+1, embed_size, padding_idx=0)


    def forward(self, q, q_len, text, text_len):
        q_embed = self.word_embedding(q).sum(1)
        q_embed /= q_len.float().unsqueeze(1)

        text_embed = self.word_embedding(text)
        text_embed = text_embed.sum(1)
        text_embed_avg = text_embed / text_len.float().unsqueeze(1)

        s = text_embed_avg.unsqueeze(1) @ q_embed.unsqueeze(2)
        s = s.view(-1)

        ## full doc as negative samples
        doc_neg = self.word_embedding.weight[1:].sum(0, keepdim=True) - text_embed
        doc_neg /= self.vocab_size

        s_neg = doc_neg.unsqueeze(1) @ q_embed.unsqueeze(2)
        s_neg = s_neg.view(-1)

        return s, s_neg


    def forward_old(self, q, q_len, text, text_len):
        q_embed = self.word_embedding(q).sum(1)
        q_embed /= q_len.float()

        text_embed = text.view(-1, text.size(2))
        text_embed = self.word_embedding(text_embed)
        text_embed = text_embed.view(text.size(0), text.size(1), text.size(2), -1)
        text_embed = text_embed.sum(2)
        text_embed /= text_len.float().unsqueeze(2)

        s = text_embed @ q_embed.unsqueeze(2)
        s = s.mean(1).squeeze()

        return s


    def loss(self, q, q_len, text_pos, pos_len, text_neg, neg_len, thres=1):
        s_pos, s_doc = self.forward(q, q_len, text_pos, pos_len)
        s_neg, _ = self.forward(q, q_len, text_neg, neg_len)

        margin1 = thres + s_neg - s_pos
        margin1 = margin1.clamp(min=0).mean()
        #margin2 = thres + s_doc - s_pos
        #margin2 = margin2.clamp(min=0).mean()

        w_norms = (self.word_embedding.weight ** 2).mean(0).sum()
        
        return margin1, w_norms

