import tensorflow as tf
def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 190496,
        embedding = None, # or the path to the pre-trained embedding
        embedding_trainable = True,
        label_size = 1, # if 1, multi-label setting
        multi_label = 11, # if >1, multi-label setting
        hidden_size = 30,
        value_depth = 30,
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # attention network parameters
        num_layers = 3,
        num_heads = 2,
        attn_dropout = 0.1,
        residual_dropout = 0.1,
        relu_dropout = 0.1,
        filter_size = 64,
        
        # convolution parameters
        kernel = [10, 30, 30], # kernel shape for tf.nn.conv1d, [filter_width, in_channels, out_channels]
        stride = 1,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # fully connected network parameters
        ffn_size = None,
        
        # learning parameters
        batch_size = 256,
        learning_rate = 0.02,
        decay = 0.4, 

        regularization = 0.005,

        lexicon_effect = None # None, 'nrc1' and float number for nrc2
        
    )
