
import pandas as pd
from sklearn.utils import resample


df= pd.read_csv('train.tsv', error_bad_lines=False, sep='\t')
df_train = df[:124848] # 80% for train
df_val = df[124848:]


df_train.to_csv('training.tsv', sep='\t', encoding='utf-8')
df_val.to_csv('validating.tsv', sep='\t', encoding='utf-8')



df_training = pd.read_csv('/home/rania/SentAnalyser/training.tsv', error_bad_lines=False, sep='\t')
df_valdating = pd.read_csv('/home/rania/SentAnalyser/validating.tsv', error_bad_lines=False, sep='\t')

#RESAMPLING - DOWNSIZING THE NEUTRAL CLASS DATA
df_train.head()
def down_sample(df):
    df_majority = df[df['Sentiment']==2] # la classe neutre 2 qui a 79582 samples
    #len(df_majority)
    others = [0,1,3,4] #the rest of classes
    keep = df['Sentiment'].isin(others)
    df_minority = df[keep]
    #len(df_minority)
    # Upsample minority class
    df_majority_upsampled = resample(df_majority,
                                     replace = True,     # sample with replacement
                                     n_samples = 54621 ,    # to match majority class
                                     random_state = 123) # reproducible results

    # Combine majority class with upsampled minority class
    df = pd.concat([df_minority, df_majority_upsampled])
    return df

df_train = down_sample(df_train)
df_val = down_sample(df_val)

df_train.to_csv('/home/rania/SentAnalyser/down_sample_training.tsv', sep='\t', encoding='utf-8')
df_val.to_csv('/home/rania/SentAnalyser/down_sample_validating.tsv', sep='\t', encoding='utf-8')
