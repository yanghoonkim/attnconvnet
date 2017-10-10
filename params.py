import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 8414,
        hidden_size = 100,
        value_depth = 100,
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # convolution parameters
        kernel = [2, 100, 100], # kernel shape for tf.nn.conv1d
        stride = 1,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # attention network parameters
        num_layers = 3,
        num_heads = 5,
        attn_dropout = 0.1,
        residual_dropout = 0.1,
        relu_dropout = 0.0,
        ffn_size = 100,
        label_size = 6,
        filter_size = 512,
        
        # learning parameters
        learning_rate = 0.7
        
    )
