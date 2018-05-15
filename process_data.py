import sys
import os
sys.path.append('submodule/')

from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd

from process import make_vocab_and_process


TRAIN_FILE = 'data/2018-E-c-En-train.txt'
DEV_FILE = 'data/2018-E-c-En-dev.txt'
TEST_FILE = 'data/2018-E-c-En-test.txt'

VOCAB = ['data/processed/vocab_ec.npy', 'data/processed/vocab_label_ec.npy']

outfile_train_data = 'data/processed/ec_train'
outfile_train_label = 'data/processed/ec_train_label'
outfile_train_length = 'data/processed/ec_train_length'

outfile_dev_data = 'data/processed/ec_dev'
outfile_dev_label = 'data/processed/ec_dev_label'
outfile_dev_length = 'data/processed/ec_dev_length'

outfile_test_data = 'data/processed/ec_test'
outfile_test_length = 'data/processed/ec_test_length'

outfile_train = [outfile_train_data, outfile_train_label, outfile_train_length]
outfile_dev = [outfile_dev_data, outfile_dev_label, outfile_dev_length]
outfile_test = [outfile_test_data, None, outfile_test_length]


# Process Train File
dataframe = pd.read_csv(TRAIN_FILE, sep='\t')

tokenize = TweetTokenizer().tokenize

tweets = map(lambda x: tokenize(x), dataframe['Tweet'])
for i, tweet in enumerate(tweets):
    for j, word in enumerate(tweet):
        if '@' in word:
            tweets[i][j] = '<MENTION>'
        if '#' in word:
            tweets[i][j] = word[1:]

labels = np.asarray(dataframe.drop(['ID', 'Tweet'], axis=1))

maxlen = max([len(sentence) for sentence in tweets])
print 'Train data maxlen = %d' %maxlen

make_vocab_and_process(tweets, labels, maxlen, outfile_train, VOCAB, if_multi_label = True)



# Process Dev File
dataframe = pd.read_csv(DEV_FILE, sep='\t')

tokenize = TweetTokenizer().tokenize

tweets = map(lambda x: tokenize(x), dataframe['Tweet'])
for i, tweet in enumerate(tweets):
    for j, word in enumerate(tweet):
        if '@' in word:
            tweets[i][j] = '<MENTION>'
        if '#' in word:
            tweets[i][j] = word[1:]

labels = np.asarray(dataframe.drop(['ID', 'Tweet'], axis=1))

maxlen = max([len(sentence) for sentence in tweets])
print 'Dev data maxlen = %d' %maxlen

make_vocab_and_process(tweets, labels, maxlen, outfile_dev, VOCAB, if_train = False, if_multi_label = True)


# Process Test File
dataframe = pd.read_csv(TEST_FILE, sep='\t')

tokenize = TweetTokenizer().tokenize

tweets = map(lambda x: tokenize(x), dataframe['Tweet'])
for i, tweet in enumerate(tweets):
    for j, word in enumerate(tweet):
        if '@' in word:
            tweets[i][j] = '<MENTION>'
        if '#' in word:
            tweets[i][j] = word[1:]

labels = np.asarray(dataframe.drop(['ID', 'Tweet'], axis=1))

maxlen = max([len(sentence) for sentence in tweets])
print 'Test data maxlen = %d' %maxlen

make_vocab_and_process(tweets, None, maxlen, outfile_test, VOCAB, if_train = False, if_multi_label = True)
