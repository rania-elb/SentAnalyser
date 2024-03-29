import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# use pathlib to load path
data_root = pathlib.Path('./')
data = pd.read_csv(data_root/'train.tsv', sep='\t')

################################
# plot averages of sentiments
numOfPhrases = 0
sumOfSent = 0
prevId = 1
currSent = dataUniq['SentenceId'][0]
currId = dataUniq['SentenceId'][0]

wantedSents = []
avgOfSents = []

wantedSents.append(data['Sentiment'][0])

for i in data['PhraseId']:
	currId = data['SentenceId'][i-1]
	if currId != prevId :
		if numOfPhrases > 0 :
			avgOfSents.append(sumOfSent/numOfPhrases)
		else:
			avgOfSents.append(0)
		
		numOfPhrases = 0
		sumOfSent = 0
		wantedSents.append(data['Sentiment'][i-1])
		
		prevId = currId
	else:
		sumOfSent = sumOfSent + data['Sentiment'][i-1]
		numOfPhrases = numOfPhrases +1

plt.plot(wantedSents[:20])
plt.plot(avgOfSents[:20])
plt.show()

#######################################
# plot number of phrases per sentiment

fig = plt.figure(figsize=(8,5))
ax = sns.barplot(x=data.Sentiment.unique(),y=data.Sentiment.value_counts());
ax.set(xlabel='Labels')
plt.show()

#################################################
# plot number of entire sentence per sentiment

# try with entire sentences only
#dataUniq = pd.read_csv(data_root/'train.tsv', sep='\t')
#prevSent= -1;
#prevId = 0;
#currSent = dataUniq['SentenceId'][0]
#
#tab = []
#
#for i in data['PhraseId']:
#	currSent = data['SentenceId'][i-1]
#	if currSent != prevSent :
#		dataUniq = dataUniq.drop(range(prevId, i-1))
#		prevSent = currSent
#		prevId = i;
#		tab.append((data['SentenceId'][i-1], data['Sentiment'][i-1]))
#
#dataUniq = dataUniq.drop(range(prevId, len(dataUniq)))

#dataUniq = []
uniqIds = []
prevSent= -1;
currSent = data['SentenceId'][0]

for i in data['PhraseId']:
	currSent = data['SentenceId'][i-1]
	if currSent != prevSent :
		#dataUniq.append([d for d in data.iloc[i-1]])
		uniqIds.append(i)
		prevSent = currSent

bx = sns.barplot(x=[d[3] for d in dataUniq],y=[d[3] for d in dataUniq].value_counts());
bx.set(xlabel='Labels')
plt.show()
