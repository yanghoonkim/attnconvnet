import tensorflow as tf
hidden =30
def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 8413,
        label_size = 6, 
        hidden_size = hidden,
        value_depth = hidden,
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # attention network parameters
        num_layers = 3,
        num_heads =1,
        attn_dropout = 0.1,
        residual_dropout = 0.1,
        relu_dropout = 0.0,
        filter_size = 100,
        
        # convolution parameters
        kernel = [5, hidden, 20], # kernel shape for tf.nn.conv1d, [filter_width, in_channels, out_channels]
        stride = 5,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # relation parameters
        relation_size = 30,

        # fully connected network parameters
        ffn_size = None,
        
        # learning parameters
        batch_size = 512,
        learning_rate = 0.1
        
    )
