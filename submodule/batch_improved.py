# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nltk
#import twkorean
import codecs
import random
import os

from milab_rnn.utils.vocab import Vocab
from milab_rnn.utils.vocab import TOKENS # ['_PAD', '_EOS', '_GO', '_UNK']

def korean_tokenizer():
    processor = twkorean.TwitterKoreanProcessor(stemming=False)
    return processor.tokenize_to_strings

def english_tokenizer():
    return nltk.word_tokenize

def basic_tokenizer():
    def tokenizer(text):
        return text.split(' ')
    return tokenizer 

class Batch_Seq2Seq(object):
    def __init__(self, source_path, target_path, max_vocab=[30000, 30000], bucket_size=None, batch_sizes=None, prepadding=False, pad_pos=(0, 1)):
        self.source_path = source_path
        self.target_path = target_path
        
        self.prepadding = prepadding
        self.pad_pos = pad_pos
        
        self.max_vocab = max_vocab
        
        if bucket_size == None:
            self.bucket_size = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
        else:
            self.bucket_size = bucket_size
            
        if batch_sizes == None:
            self.batch_sizes = [512, 512, 256, 256, 128, 128]
        else:
            self.batch_sizes = batch_sizes
            
        self.source_vocab = Vocab()
        self.target_vocab = Vocab()
        
    def _read_raw_data(self, tokenizers=[None, None], make_vocab=True, insert_go = (0,1), which_data = 'train', vocab_path = None):
        '''Read & tokenize raw data
        This function will be used inside preprocess function
        
        Args : 
            tokenizers : each for source language and target language
            insert_go : each for source language and target language
            vocab_path : [soruce_vocab_path, target_vocab_path]
            
        Return:
            list of numeralized sentences with numpy array format 
        '''
        if tokenizers[0] == None:
            tokenizers[0] = basic_tokenizer()
        if tokenizers[1] == None:
            tokenizers[1] = basic_tokenizer()
            
        print('Open & Tokenize source data in ' + self.source_path + '......')
        f = open(self.source_path, 'r')
        source_lines = [[word.lower() for word in tokenizers[0](unicode(line.strip(), 'utf-8'))] for line in f.readlines()]
        f.close()
        print('......Tokenization completed.')
        print('(' + str(len(source_lines)) + ' lines in ' + self.source_path + '.)\n')
        
        print('Open & Tokenize target data in ' + self.target_path + '......')
        f = open(self.target_path, 'r')
        target_lines = [[word.lower() for word in tokenizers[1](unicode(line.strip(), 'utf-8'))] for line in f.readlines()]
        f.close()
        print('......Tokenization completed.')
        print('(' + str(len(target_lines)) + ' lines in ' + self.target_path + '.)\n')
        
        if make_vocab:
            if(which_data == 'train'):
                print('Make vocab for each source & target languages' + '......')
                for line in source_lines:
                    for word in line:
                        self.source_vocab.add_word(word)

                for line in target_lines:
                    for word in line:
                        self.target_vocab.add_word(word)
            
                self.source_vocab.word_count()
                self.target_vocab.word_count()

                self.source_vocab.cut_by_freq(self.max_vocab[0])
                self.target_vocab.cut_by_freq(self.max_vocab[1])
                self.source_vocab.save(vocab_path[0])
                self.target_vocab.save(vocab_path[1])
                print('......Completed: each vocab has ' + str(self.source_vocab.size) + ' words, ' + str(self.target_vocab.size) + ' words.\n')
            else: # load vocab which is created with train set
                print('Load vocab ...')
                self.source_vocab.load(vocab_path[0])
                self.target_vocab.load(vocab_path[1])
                
        
        print('Data Numeralization......', end = ' ')
        source_data = []
        target_data = []
        for idx in xrange(len(source_lines)):
            if len(source_lines[idx]) == 0 or len(target_lines[idx]) == 0:
                continue
            source_integer_sentence = []
            target_integer_sentence = []
            
            # encode source lines
            if insert_go[0]:
                source_integer_sentence.append(self.source_vocab.word2index[TOKENS[2]]) # add _GO symbol
            for word in source_lines[idx]:
                if word not in self.source_vocab.word2index:
                    source_integer_sentence.append(self.source_vocab.word2index[TOKENS[3]]) # add _UNK symbol
                else:
                    source_integer_sentence.append(self.source_vocab.word2index[word])
            source_integer_sentence.append(self.source_vocab.word2index[TOKENS[1]]) # add _EOS symbol
            source_data.append(np.asarray(source_integer_sentence, dtype=np.int32))
            
            # encode target lines
            if insert_go[1]:
                target_integer_sentence.append(self.target_vocab.word2index[TOKENS[2]]) # add _GO symbol
            for word in target_lines[idx]:
                if word not in self.target_vocab.word2index:
                    target_integer_sentence.append(self.target_vocab.word2index[TOKENS[3]]) # add _UNK symbol
                else:
                    target_integer_sentence.append(self.target_vocab.word2index[word])
            target_integer_sentence.append(self.target_vocab.word2index[TOKENS[1]]) # add _EOS symbol
            target_data.append(np.asarray(target_integer_sentence, dtype=np.int32))
        print('Completed.\n')
        
        return source_data, target_data   
    
    def _pile_buckets(self, source_sentences, target_sentences):
        '''assign buckets to each line of sentences
        This function will be used inside preprocess function
        
        '''
        print('Put sentences pairs into designated buckets......', end = ' ')
        buckets = []
        for _ in xrange(len(self.bucket_size)):
            buckets.append([]) # looks like [[], [], [], [], [], []] 
        for (source_sentence, target_sentence) in zip(source_sentences, target_sentences):
            source_len = len(source_sentence)
            target_len = len(target_sentence)
            idxex = -1
            for i in xrange(len(self.bucket_size)):
                if (source_len <= self.bucket_size[i][0]) and (target_len <= self.bucket_size[i][1]):
                    idxex = i
                    #print(idxex)
                    break
            if idxex == -1: # if the sentence length is out of range
                continue
            
            if (len(source_sentence) <= 1) or (len(target_sentence) <= 1):
                continue
            
            buckets[i].append((source_sentence, target_sentence))
            
        print('Completed.\n')
        
        return buckets
    
    
    def _pad_data(self, bucket_idx, pair_sentences, swap_side=False, pad_pos=(1, 1)):
        '''add padding to one given bucket
        
        Args : 
        swap_size(for dual learning) : swapping the source/target data
        pad_pos : for example: (0, 1) means that <padding position>source sentence + target_sentence<pading position>
        
        '''
        bucket_size = self.bucket_size[bucket_idx]
        
        padded_sources = []
        padded_targets = []
        target_masks = []
        if pair_sentences != []:
            src_sentences, trg_sentences = zip(*pair_sentences)
            if swap_side == False: # if not dual learning
                source_max_length = bucket_size[0]
                target_max_length = bucket_size[1] + 1

                source_sentences = src_sentences
                target_sentences = trg_sentences
            else: # if dual learning
                source_max_length = bucket_size[1]
                target_max_length = bucket_size[0] + 1

                source_sentences = trg_sentences
                target_sentences = src_sentences


            for (source_sentence, target_sentence) in pair_sentences: # for each sentence             
                #source_sentence = source_sentence[1:] # remove _GO symbol from source sentences

                source_pad = (source_max_length - source_sentence.shape[0], 0)
                target_pad = (0, target_max_length - target_sentence.shape[0])

                if pad_pos[0] == 1:
                    source_pad = source_pad[::-1]
                if pad_pos[1] == 0:
                    target_pad = target_pad[::-1]

                padded_sources.append(np.pad(source_sentence, source_pad, 'constant', constant_values=0)) # word2index('_PAD') = 0
                padded_targets.append(np.pad(target_sentence, target_pad, 'constant', constant_values=0))

                mask = np.ones(shape=padded_targets[-1].shape, dtype=np.float32) 
                mask[np.where(padded_targets[-1] == 0)] = 0
                target_masks.append(mask)
            
        return np.asarray(padded_sources), np.asarray(padded_targets), np.asarray(target_masks) 
        # why return np.asarray()? : when input the data to tensorflow model, the data should looks like length T list of [batch_size]. when using asarray, it is easy to convert from length batch list of [sequence_length] to length T list of [batch_size] with command data[:,i] for i in range(sequence_length)

    
    def preprocess(self, tokenizers=[None, None], make_vocab=True, which_data = 'train', vocab_path = None):
        print("<<Preprocessing data in all buckets>>\n")
        self.source_data, self.target_data = self._read_raw_data(tokenizers=tokenizers, make_vocab=make_vocab, which_data = which_data, vocab_path = vocab_path)
        self.buckets = self._pile_buckets(self.source_data, self.target_data)
        if self.prepadding:
            print('pre-padding data in all buckets ......', end = ' ')
            for i, bucket in enumerate(self.buckets):
                padded_sources, padded_targets, padded_masks = self._pad_data(i, bucket, pad_pos=self.pad_pos)
                self.buckets[i] = zip(*(padded_sources, padded_targets, padded_masks))
            print("Completed.\n")
        print('<<Preprocessing Completed>>\n')
            
            
    def next_batch(self, do_shuffle=True):
        '''just pick one batch from buckets
        
        Args : 
        
        '''
        num_bucket = len(self.buckets)
        idx_for_each_bucket = [0] * num_bucket
        is_bucket_done = [False] * num_bucket
        bucket_idx = 0
        buckets = []
        
        if do_shuffle: 
            for i in xrange(num_bucket):
                buckets.append(list(self.buckets[i]))
                random.shuffle(buckets[i])        
        else: # 그냥 이퀄 하면 되는데?
            for i in xrange(num_bucket):
                buckets.append(list(self.buckets[i])) # why list? unnacessary

        def get_bucket_idx(num_bucket, is_bucket_done):
            '''randomly pick the bucket which is remained
            '''
            if False in is_bucket_done: # if some buckets are still remained
                chk = True
                while chk == True: # until find out remained bucket
                    bucket_idx = random.randrange(0, num_bucket) # why? overlapping problem
                    chk = is_bucket_done[bucket_idx]

                return bucket_idx
            else:
                return -1
            
        while True:
            bucket_idx = get_bucket_idx(num_bucket, is_bucket_done)
            
            if len(buckets[bucket_idx]) == 0: # error handling for blank bucket
                is_bucket_done[bucket_idx] = True
                continue
            
            if bucket_idx == -1:
                raise StopIteration
            
                
            idx_for_each_bucket[bucket_idx] += 1 # -th batch sementation
            
            data_idx_start = (idx_for_each_bucket[bucket_idx] - 1) * self.batch_sizes[bucket_idx]
            data_idx_end = idx_for_each_bucket[bucket_idx] * self.batch_sizes[bucket_idx]
            is_bucket_nonempty = int(len(buckets[bucket_idx]) / self.batch_sizes[bucket_idx]) > idx_for_each_bucket[bucket_idx] - 1 # modify
           
            if not self.prepadding:
                if is_bucket_nonempty:
                    source_sentences = [buckets[bucket_idx][i][0] for i in xrange(data_idx_start, data_idx_end)]
                    target_sentences = [buckets[bucket_idx][i][1] for i in xrange(data_idx_start, data_idx_end)]
                    yield  bucket_idx, source_sentences, target_sentences                
                else:
                    is_bucket_done[bucket_idx] = True
                    source_sentences = [buckets[bucket_idx][i][0] for i in xrange(data_idx_start, len(buckets[bucket_idx]))]
                    target_sentences = [buckets[bucket_idx][i][1] for i in xrange(data_idx_start, len(buckets[bucket_idx]))]
                    yield bucket_idx, source_sentences, target_sentences
            else:
                if is_bucket_nonempty:
                    source_sentences = [buckets[bucket_idx][i][0] for i in xrange(data_idx_start, data_idx_end)]
                    target_sentences = [buckets[bucket_idx][i][1] for i in xrange(data_idx_start, data_idx_end)]
                    target_masks = [buckets[bucket_idx][i][2] for i in xrange(data_idx_start, data_idx_end)]
                    yield bucket_idx, np.asarray(source_sentences), np.asarray(target_sentences), np.asarray(target_masks)
                else:                
                    is_bucket_done[bucket_idx] = True
                    source_sentences = [buckets[bucket_idx][i][0] for i in xrange(data_idx_start, len(buckets[bucket_idx]))]
                    target_sentences = [buckets[bucket_idx][i][1] for i in xrange(data_idx_start, len(buckets[bucket_idx]))]
                    target_masks = [buckets[bucket_idx][i][2] for i in xrange(data_idx_start, len(buckets[bucket_idx]))]
                    yield bucket_idx, np.asarray(source_sentences), np.asarray(target_sentences), np.asarray(target_masks)