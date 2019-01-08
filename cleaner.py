import pandas as pd
from nltk.corpus import stopwords

df = pd.read_csv('train2.tsv', sep='\t')

stopWords = set(stopwords.words('english'))

df = df.apply(lambda x: x.astype(str).str.lower())
df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopWords)]))

df.to_csv("train2.tsv", sep='\t')
