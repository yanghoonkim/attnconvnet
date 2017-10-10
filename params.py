import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 8414,
        hidden_size = 300,
        value_depth = 300, # equal to hidden_size, in_channels
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # convolution parameters
        kernel = [2, 300, 100], # kernel shape for tf.nn.conv1d [filter_width, in_channels, out_channels]
        stride = 1,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # attention network parameters
        num_layers = 3,
        num_heads = 2, # must devide hidden_size
        attn_dropout = 0.1,
        residual_dropout = 0.1,
        relu_dropout = 0.0,
        ffn_size = None,
        label_size = 6,
        filter_size = 512,
        
        # learning parameters
        learning_rate = 0.7
        
    )
