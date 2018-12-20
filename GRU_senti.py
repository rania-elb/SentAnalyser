
import os, sys
import re
import string
import pathlib
import random
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score



df = pd.read_csv('train.tsv', error_bad_lines=False, sep='\t')
#df.shape
#df.head()

#df.Sentiment.value_counts()

"""
fig = plt.figure(figsize=(8,5))
ax = sns.barplot(x=df.Sentiment.unique(),y=df.Sentiment.value_counts());
ax.set(xlabel='Labels');

"""

"""

 Build Vocabulary and tokenize

"""



# load spacy tokenizer
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])

nlp(df.Phrase.values[0])

# remove leading and trailing spaces
df['SentimentText'] = df.Phrase.progress_apply(lambda x: x.strip())


words = Counter()
for sent in tqdm_notebook(df.Phrase.values):
    words.update(w.text.lower() for w in nlp(sent))
#len(words)
#words.most_common(20)



words = sorted(words, key=words.get, reverse=True)
#words[:20]

words = ['_PAD','_UNK'] + words
#words[:10]


word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}
def indexer(s): return [word2idx[w.text.lower()] for w in nlp(s)]
"""
Tokenize and calculate tweet length
"""

df['sentimentidx'] = df.Phrase.apply(indexer)
df.head()

"""
 Padded dataset and dataloader
"""

# subclass the custom dataset class with torch.utils.data.Dataset
# implement __len__ and __getitem__ function
class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=200):
        self.maxlen = maxlen
        self.df = pd.read_csv(df_path, error_bad_lines=False, sep='\t')
        self.df['Phrase'] = self.df.Phrase.apply(lambda x: x.strip())
        print('Indexing...')
        self.df['sentimentidx'] = self.df.Phrase.progress_apply(indexer)
        print('Calculating lengths')
        self.df['lengths'] = self.df.sentimentidx.progress_apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
        print('Padding')
        self.df['sentimentpadded'] = self.df.sentimentidx.progress_apply(self.pad_data)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = self.df.sentimentpadded[idx]
        lens = self.df.lengths[idx]
        y = self.df.Sentiment[idx]
        return X,y,lens

    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded

# create instance of custom dataset
ds = VectorizeData('train.tsv')

#dl = DataLoader(dataset=ds, batch_size=3)
#print('Total batches', len(dl))
#it = iter(dl)
#xs,ys,lens =  next(it)
#print(type(xs))
#print(xs)



#print('Labels:',ys)
#print('Lengths:',lens)


#initiliazation
vocab_size = len(words)
embedding_dim = 4
n_hidden = 5
n_out = 5


class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        self.vocab_size,self.embedding_dim,self.n_hidden,self.n_out = vocab_size, embedding_dim, n_hidden, n_out
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden)
        self.out = nn.Linear(self.n_hidden, self.n_out)

    def forward(self, seq, lengths, gpu=True):
        print('Sequence shape',seq.shape)
        print('Lengths',lengths)
        bs = seq.size(1) # batch size
        print('batch size', bs)
        self.h = self.init_hidden(bs, gpu) # initialize hidden state of GRU
        print('Inititial hidden state shape', self.h.shape)
        embs = self.emb(seq)
        embs = pack_padded_sequence(embs, lengths) # unpad
        gru_out, self.h = self.gru(embs, self.h) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        gru_out, seq_lengths = pad_packed_sequence(gru_out) # pad the sequence to the max length in the batch
        print('GRU output(all timesteps)', gru_out.shape)
        print(gru_out)
        print('GRU last timestep output')
        print(gru_out[-1])
        print('Last hidden state', self.h)
        # since it is as classification problem, we will grab the last hidden state
        outp = self.out(self.h[-1]) # self.h[-1] contains hidden state of last timestep
        return F.log_softmax(outp, dim=-1)

    def init_hidden(self, batch_size, gpu):
        if gpu: return Variable(torch.zeros((1,batch_size,self.n_hidden)).cuda())
        else: return Variable(torch.zeros((1,batch_size,self.n_hidden)))

m = SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out)
#print(m)

#Function to sort the batch according to Phrase lengths
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)



dl = DataLoader(ds, batch_size=3)
it = iter(dl)
xs,ys,lens =  next(it)


xs,ys,lens = sort_batch(xs,ys,lens)
#outp = m(xs,lens.cpu().numpy(), gpu=False) # last non zero values from gru is same as hidden output by gru
"""
TRainig the model
"""
def replaced(sequence, old, new):
    return (new if x == old else x for x in sequence)
def fit(model, train_dl, val_dl, loss_fn, opt, epochs=3):
    num_batch = len(train_dl)
    for epoch in tnrange(epochs):
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0

        if val_dl:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0

        t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
        for X,y, lengths in t:
            t.set_description(f'Epoch {epoch}')
            X,y,lengths = sort_batch(X,y,lengths)
            X = Variable(X.cuda())
            y = Variable(y.cuda())
            lengths = list(replaced(lengths, 0, 1))
            #lengths = lengths.numpy()

            opt.zero_grad()
            pred = model(X, lengths, gpu=True)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            t.set_postfix(loss=loss.item())
            pred_idx = torch.max(pred, dim=1)[1]

            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred_idx.cpu().data.numpy())
            total_loss_train += loss.item()

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_loss = total_loss_train/len(train_dl)
        print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')

        if val_dl:
            for X,y,lengths in tqdm_notebook(valdl, leave=False):
                X, y,lengths = sort_batch(X,y,lengths)
                X = Variable(X.cuda())
                y = Variable(y.cuda())
                pred = model(X, lengths.numpy())
                loss = loss_fn(pred, y)
                pred_idx = torch.max(pred, 1)[1]
                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(pred_idx.cpu().data.numpy())
                total_loss_val += loss.item()
            valacc = accuracy_score(y_true_val, y_pred_val)
            valloss = total_loss_val/len(valdl)
            print(f'Val loss: {valloss} acc: {valacc}')


train_dl = DataLoader(ds, batch_size=50)
m = SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out).cuda()
opt = optim.Adam(m.parameters(), 1e-2)

fit(model=m, train_dl=train_dl, val_dl=None, loss_fn=F.nll_loss, opt=opt, epochs=4)
torch.save(m, 'simple_gru_model.pt')
