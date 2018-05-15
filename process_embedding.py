import numpy as np

# Load & process GloVe
data_dir = 'data/'
glove = 'glove.840B.300d'

print 'Reading Glove...'
f = open(data_dir + glove + '.txt')
lines = f.readlines()
f.close()

print 'Processing Glove...'
embedding = dict()
for line in lines:
    splited = line.split()
    embedding[splited[0]] = map(float, splited[1:])

# Save glove as dic file
np.save(data_dir + 'processed/' + glove + '.dic', embedding)

vocab = np.load('data/processed/vocab_ec.npy').item()

print 'Producing pre-trained embedding...'
embedding_vocab =  np.random.ranf((len(vocab), 300)) -  np.random.ranf((len(vocab), 300))
embedding_vocab[1] = 0.0 # vocab['<PAD>'] = 1

unk_num = 0
for word, idx in vocab.items():
    if word in embedding:
        embedding_vocab[idx] = embedding[word]
    else:
        unk_num += 1


np.save('data/processed/glove_embedding.npy', embedding_vocab)

# check how many unknown words
print 'vocab size : %d' %len(embedding_vocab)
print 'unknown word size : %d' %unk_num