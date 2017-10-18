
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[ ]:

import numpy as np
import codecs
import os


# In[ ]:

TOKENS = ['_PAD', '_EOS', '_GO', '_UNK']


# In[ ]:

class Vocab(object):
    def __init__(self):
        self.word2index = {}
        self.index2word = []
        self.word_freq = []
        
        map(self.add_word, TOKENS)
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.index2word.append(word)
            self.word_freq.append(1)
        else:
            self.word_freq[self.word2index[word]] += 1
            
    def word_count(self):
        print('The size of whole vocabulary ' + type(self).__name__ + ' is : %d'%len(self.index2word))
            
    def cut_by_freq(self, num_words):        
        self.word_freq = np.asarray(self.word_freq)
        indices = np.argsort(self.word_freq[len(TOKENS):])[-num_words:][::-1] + len(TOKENS)        
        self.word_freq = list(self.word_freq[:len(TOKENS)]) + list(self.word_freq[indices])
        self.index2word = self.index2word[:len(TOKENS)] + [self.index2word[ind] for ind in indices]
        indices_TOKENS = [self.word2index[token] for token in TOKENS]
        self.index2word = list(self.index2word)
        self.word2index = {self.index2word[ind]:ind for ind in xrange(len(self.index2word))}
    
    def save(self, filename):
        f = codecs.open(filename, 'w', 'utf-8')
        for i in range(len(self.index2word)):
            f.write(unicode(self.index2word[i]))
            f.write('\n')
        f.close()
        
    def load(self, filename):
        if os.path.exists(filename):
            f = codecs.open(filename, 'r', 'utf-8')
            words = [line.rstrip() for line in f.readlines()]
            for word in words:
                self.add_word(word)
            f.close()
            
            return False # make_vocab = False
        else :
            return True # make_vocab = True
            
            
        
    @property
    def size(self):
        return len(self.word2index)

