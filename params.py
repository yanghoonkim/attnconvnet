import tensorflow as tf
def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 8680,
        embedding = None,
        embedding_trainable = True,
        label_size = 6,
        multi_label = 1, # if >1, multi-label setting
        hidden_size = 50,
        value_depth = 50,
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # attention network parameters
        num_layers = 1,
        num_heads = 10,
        attn_dropout = 0.1,
        residual_dropout = 0.1,
        relu_dropout = 0.1,
        filter_size = 128,
        
        # convolution parameters
        kernel = [10, 50, 30], # kernel shape for tf.nn.conv1d, [filter_width, in_channels, out_channels]
        stride = 1,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # fully connected network parameters
        ffn_size = None,
        
        # learning parameters
        batch_size = 20,
        learning_rate = 0.02
        
    )

def trec_params():
    hparams = basic_params()
    hparams.embedding = 'data/trec/processed/glove840b_trec_vocab300.npy'
    return hparams

def sst5_params():
    hparams = basic_params()
    hparams.voca_size = 16560
    hparams.embedding = 'data/sst/processed_git/glove840b_sst5_vocab300.npy'
    hparams.embedding_trainable = True
    hparams.label_size = 5
    hparams.filter_size = 100
    hparams.batch_size = 150
    hparams.learning_rate = 0.03
    return hparams

def sem4_params():
    hidden = 20
    hparams = basic_params()
    hparams.voca_size = 4674
    hparams.embedding = 'data/semeval/processed/glove840b_semeval1_4_vocab300.npy'
    hparams.embedding_trainable = False
    hparams.label_size = 7
    hparams.hidden_size = hidden
    hparams.value_depth = hidden
    hparams.num_layers = 3
    hparams.num_heads = 2
    hparams.filter_size = 64
    hparams.kernel = [10, hidden, 30]
    hparams.batch_size = 20
    hparams.learning_rate = 0.02
    return hparams

def sem5_params(): # 52.3
    hidden = 30
    hparams = basic_params()
    hparams.voca_size = 13376
    hparams.embedding = 'data/semeval/processed/glove840b_semeval1_5_vocab300.npy'
    hparams.embedding_trainable = False
    hparams.label_size = 1
    hparams.multi_label = 11
    hparams.hidden_size = hidden
    hparams.value_depth = hidden
    hparams.num_layers = 3
    hparams.num_heads = 2
    hparams.filter_size = 64
    hparams.kernel = [10, hidden, 30]
    hparams.batch_size = 20
    hparams.learning_rate = 0.02
    return hparams


def sem5_params_ignore_bias():
    hparams = sem5_params()
    #hparams.voca_size = 19140
    #hparams.embedding = 'data/semeval/processed/glove840b_semeval1_5_vocab300_all_with_zero.npy'
    hparams.voca_size = 190467
    hparams.embedding = 'data/semeval/processed/glove840b_semeval1_5_vocab300_unlabel.npy'
    hparams.embedding_trainable = False 
    return hparams

def sem5_emo():
    hparams = sem5_params()
    hparams.voca_size = 190496
    hparams.embedding = 'data/semeval/processed/glove840b_semeval1_5_vocab300_emo_unlabel.npy'
    hparams.add_hparam('regularization', 0.0005)
    hparams.learning_rate = 0.001
    return hparams
