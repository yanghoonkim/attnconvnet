import tensorflow as tf
hidden = 50
def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 8414,
        label_size = 6, 
        hidden_size = 50,
        value_depth = 50,
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # attention network parameters
        num_layers = 1,
        num_heads =5,
        attn_dropout = 0.1,
        residual_dropout = 0.1,
        relu_dropout = 0.0,
        filter_size = 256,
        
        # convolution parameters
        kernel = [5, 50, 30], # kernel shape for tf.nn.conv1d, [filter_width, in_channels, out_channels]
        stride = 5,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # fully connected network parameters
        ffn_size = None,
        
        # learning parameters
        batch_size = 512,
        learning_rate = 0.1
        
    )

# trec: up to 0.876, maxlength20, voca 8414
def basic_1():
    return hparams
hidden = 50
def test_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 8680,
        label_size = 6, 
        hidden_size = hidden,
        value_depth = hidden,
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # attention network parameters
        num_layers = 1,
        num_heads =10,
        attn_dropout = 0.0,
        residual_dropout = 0.0,
        relu_dropout = 0.0,
        filter_size = 128,
        
        # convolution parameters
        kernel = [10, hidden, 30], # kernel shape for tf.nn.conv1d, [filter_width, in_channels, out_channels]
        stride = 1,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # fully connected network parameters
        ffn_size = None,
        
        # learning parameters
        batch_size = 20,
        learning_rate = 0.1
        
    )
