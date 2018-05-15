import numpy as np

def make_vocab_and_process(sentences, labels, maxlen, outfile, vocab, if_train = True, if_case = False, if_multi_label = False):
    '''
    args:
        sentences : looks like : [['I', 'have', 'a', 'dream'], ['I', 'love', 'you']]
        labels : looks like : ['good', 'bad'] or batch of binary multi label
        if_case : if True, case-sensitive
    return:
        vocab : [path of vocab_data, path of vocab_label]
        processed_sentences : numpy array, processed with vocab
        processed_labels
        sentence_length
    '''
    
    if if_train:
        # make and save new vocabulary
        vocab_data = dict()
        vocab_data['<UNK>'] = 0
        vocab_data['<PAD>'] = 1
        idx = 2
    
        vocab_label = dict()
    
        # make vocab for data
        for i, line in enumerate(sentences):
            if len(line) <=maxlen:
                for word in line:
                    if if_case:
                        if word not in vocab_data:
                            vocab_data[word] = idx
                            idx += 1
                        
                    else:
                        if word.lower() not in vocab_data:
                            vocab_data[word.lower()] = idx
                            idx += 1
        print 'data vocab size : %d...'%len(vocab_data)
        np.save(vocab[0], vocab_data)

        # make vocab for label
        if not if_multi_label:
            idx = 0
            for lab in labels:
                if lab not in vocab_label:
                    vocab_label[lab] = idx
                    idx += 1
            print 'label vocab size : %d...'%len(vocab_label) 
            np.save(vocab[1], vocab_label)
    
    # if not train
    else:
        vocab_data = np.load(vocab[0]).item()
        if not if_multi_label:
            vocab_label = np.load(vocab[1]).item()
        
    
    processed_sentences = list()
    processed_labels = list()
    sentence_length = list()
    
    for i, line in enumerate(sentences):
        if len(line) <= maxlen:
            processed_sentences.append([])
            processed_labels.append([])
            sentence_length.append([])
        
            for word in line:
                if if_case:
                    if word not in vocab_data:
                        processed_sentences[-1].append(vocab_data['<UNK>'])
                    else:
                        processed_sentences[-1].append(vocab_data[word])
                else:
                    if word.lower() not in vocab_data:
                        processed_sentences[-1].append(vocab_data['<UNK>'])
                    else:    
                        processed_sentences[-1].append(vocab_data[word.lower()])
            if not if_multi_label and (labels != None):
                processed_labels[-1] = vocab_label[labels[i]]
            sentence_length[-1] = len(processed_sentences[-1])
    
    # add padding
    for idx, line in enumerate(processed_sentences):
        processed_sentences[idx] += (maxlen-len(line)) * [vocab_data['<PAD>']]
    
    # convert to nparray
    processed_sentences = np.asarray(processed_sentences, dtype = np.int)
    if not if_multi_label and (labels != None):
        processed_labels = np.asarray(processed_labels, dtype = np.float)
    else :
        processed_labels = np.asarray(labels)
    sentence_length = np.asarray(sentence_length, dtype = np.int)
    
    np.save(outfile[0], processed_sentences)
    if labels is not None:
        np.save(outfile[1], processed_labels)
    np.save(outfile[2], sentence_length)
    print 'processing complete\n'
    
