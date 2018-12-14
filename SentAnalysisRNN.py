import pandas as pd

df = pd.read_csv('train.tsv', sep='\t')

from sklearn.feature_extraction.text import CountVectorizer

df = df [:1000] #pour une charge de travail moindre
vectorizer = CountVectorizer(binary=True)
text_data = df["Phrase"]
text_data = vectorizer.fit_transform(text_data)
# text_data devient un vecteur creux n.m : n phrases, m mots dans le vocabulaire.

import torch

text_tensor = torch.from_numpy(text_data.todense()).float()
# text_tensor est une matrice n.m

#ajout du cas où on ne connait pas le mot
voca = vectorizer.get_feature_names() + list(['<unk>'])
#ajout d'un zéro à la fin de chaque tensor
unk = torch.zeros([text_tensor.shape[0],1], dtype=torch.float)
text_tensor = torch.cat((text_tensor, unk), 1)
#text_tensor passe à n.(m+1)

label_data = pd.get_dummies(df["Sentiment"])
# get_dummies : donne une matrice n.c où c est le nombre de valeurs différentes pour Sentiment, soit 5 (de 0 à 4)

label_tensor = torch.tensor(label_data.values, dtype=torch.long)
# contient la matrice de label_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cuda object

import torch.nn as nn

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		
		self.hidden_size = hidden_size
		
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
	
	def forward(self, input, hidden):
		combined = torch.cat((input.view(1, len(voca)), hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden
	
	def initHidden(self):
		return torch.zeros(1, self.hidden_size, dtype=torch.float)


input_dim = len(voca) #nombre de mots, dimension de départ ; compte le mot inconnu <unk>
hidden_dim = 100 # arbitraire ?
output_dim = 5 #une dimension par sentiment

rnn = RNN(input_dim, hidden_dim, output_dim)

# hidden = torch.zeros(1000, hidden_dim, dtype=torch.float)

# output, next_hidden = rnn(text_tensor, hidden)

criterion = nn.NLLLoss()

learning_rate = 0.005 #unité ?

def train(category_tensor, line_tensor):
	hidden = rnn.initHidden() #créée une matrice nulle
	
	rnn.zero_grad()
	
	#print(rnn.parameters()[0].grad)
	#     print(line_tensor.size(), hidden.size())
	#     for i in range(line_tensor.size()[0]):    
	#         print(line_tensor[i].size())
	output, hidden = rnn(line_tensor, hidden)
	
	loss = criterion(output, torch.tensor([category_tensor]))
	loss.backward()
	
	# Add parameters' gradients to their values, multiplied by learning rate
	for p in rnn.parameters():
		print(type(p.grad))
		p.add_(-learning_rate, p.grad.data)
	
	return output, loss.item()

current_loss = 0

########
# Randomization des états initiaux


# Find letter index from voca, e.g. "a" = 0
def letterToIndex(letter):
    return .index(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, input_dim)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

import random

all_categories = [0,1,2,3,4]
#category_lines = {0:['null', 'horrible'], 1:['idiotic', 'negative'], 2 : ['normal'], 3 : ['cool'], 4: ['super'] }

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

######

# for label, text in zip(label_tensor, text_tensor):
for i in range(text_tensor.size()[0]):
	output, loss = train(category_tensor, line_tensor)
	current_loss += loss

