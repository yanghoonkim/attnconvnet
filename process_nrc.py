import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer

tokenize = TweetTokenizer().tokenize
lemmatizer = nltk.stem.WordNetLemmatizer()

LEXICON = 'data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

TRAIN_FILE = 'data/2018-E-c-En-train.txt'
DEV_FILE = 'data/2018-E-c-En-dev.txt'
TEST_FILE = 'data/2018-E-c-En-test.txt'

TRAIN_OUT = 'data/processed/nrc_train.npy'
DEV_OUT = 'data/processed/nrc_dev.npy'
TEST_OUT = 'data/processed/nrc_test.npy'

vocab_senti = {'anger':0, 'anticipation':1, 'disgust':2, 'fear':3, 'joy':4, 'love':5, 'positive' :6, 'negative':7, 'sadness':8, 'surprise':9, 'trust':10}


def remove_at_hashtag(tweets):
    for i, tweet in enumerate(tweets):
        for j, word in enumerate(tweet):
            if '@' in word:
                tweets[i][j] = '<MENTION>'
            if '#' in word:
                tweets[i][j] = word[1:]
    return tweets

def lemmatize_and_lower(list_of_tokens):
    return [lemmatizer.lemmatize(t.lower()) for t  in list_of_tokens]


def extract_sentiment(df, lexicon):
    
    tweet = map(lambda x: tokenize(x), df['Tweet'])
    tweet = remove_at_hashtag(tweet)
    lemma = map(lemmatize_and_lower, tweet)
    senti_list = list()

    for i, line in enumerate(lemma):
        senti_list.append([0]*11)
        for j, token in enumerate(line):
            overlap = list()
            for k, word in enumerate(lexicon['word']):
                if token == word:
                    sentiment = lexicon['sentiment'].iloc[k]
                    if sentiment not in overlap:
                        overlap.append(sentiment)
                        senti_list[-1][vocab_senti[sentiment]] += 1
        if i%20 == 0:
            print 'Processing %d-th...line\r'%i,
    print 'processing complete\n'
    return np.asarray(senti_list)


df_lexicon = pd.read_csv(LEXICON, sep = '\t', names = ['word', 'sentiment','association'])
nrc = df_lexicon[df_lexicon['association'] != 0]
nrc = nrc.drop(['association'], axis = 1)

# Process Train data
print 'Processing train data...'
df_train = pd.read_csv(TRAIN_FILE, sep = '\t')
train_senti = extract_sentiment(df_train, nrc)
np.save(TRAIN_OUT, train_senti)

# Process Dev data
print 'Processing dev data...'
df_dev = pd.read_csv(DEV_FILE, sep = '\t')
dev_senti = extract_sentiment(df_dev, nrc)
np.save(DEV_OUT, dev_senti)

# Process Test data
print 'Processing test data...'
df_test = pd.read_csv(TEST_FILE, sep = '\t')
test_senti = extract_sentiment(df_test, nrc)
np.save(TEST_OUT, test_senti)