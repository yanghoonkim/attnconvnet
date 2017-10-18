import sys
sys.path.append('submodule/')

import numpy as np
import tensorflow as tf

import common_attention as ca
import common_layers as cl

def attn_net(features, labels, mode, params):
    hidden_size = params['hidden_size']
    voca_size = params['voca_size']
    bucket_sizes = params['bucket_sizes']
    
    
    def residual_fn(x, y):
        return cl.layer_norm(x + tf.nn.dropout(
            y, 1.0 - params['residual_dropout']))
    
    def embed_op(inputs, params):
        embedding = tf.get_variable('embedding', [params['voca_size'], params['hidden_size']], dtype = params['dtype'])
        return tf.nn.embedding_lookup(embedding, inputs)

    def conv_op(embd_inp, params):
        fltr = tf.get_variable('conv_fltr', params['kernel'], params['dtype'])
        convout = tf.nn.conv1d(embd_inp, fltr, params['stride'], params['conv_pad'])
        return convout

    def ffn_op(x, params):
        g = lambda x, y, z : tf.tanh(tf.matmul(x, y) + z)

        out = x
        if type(params['ffn_size']) == None:
            layers = [params['label_size']]
        elif type(params['ffn_size']) == int:
            layers = [params['ffn_size'], params['label_size']]
        else: # list
            layers = params['ffn_size'] + [params['label_size']]

        w_ffn = list()
        b_ffn = list()
        for i, layer in enumerate(layers):
            if i==0:
                w_ffn.append(tf.get_variable('w_ffn{}'.format(i), [params['hidden_size'], layer], params['dtype']))
                b_ffn.append(tf.get_variable('b_ffn{}'.format(i), [layer], params['dtype']))
            else:
                w_ffn.append(tf.get_variable('w_ffn{}'.format(i), [layers[i-1], layer], params['dtype']))
                b_ffn.append(tf.get_variable('b_ffn{}'.format(i), [layer], params['dtype']))
            out = g(out, w_ffn[i], b_ffn[i])

        return out
    
    def transformer_ffn_layer(x, params):
        """Feed-forward layer in the transformer.
        Args:
            x: a Tensor of shape [batch_size, length, hparams.hidden_size]
            hparams: hyperparmeters for model
        Returns:
            a Tensor of shape [batch_size, length, hparams.hidden_size]
        """
        return cl.conv_hidden_relu(
            x,
            params['filter_size'],
            params['hidden_size'],
            dropout=params['relu_dropout'])

    inputs = features['x']
    
    # raw input to embedded input of shape [batch, length, embedding_size]
    embd_inp = embed_op(inputs, params)

    x = embd_inp
    

    for layer in xrange(params['num_layers']):
        x = residual_fn(
        x,
        ca.multihead_attention(
                query_antecedent = x,
                memory_antecedent = None, 
                bias = None,
                total_key_depth = params['hidden_size'],
                total_value_depth = params['value_depth'],
                output_depth = params['hidden_size'],
                num_heads = params['num_heads'],
                dropout_rate = params['attn_dropout'],
                summaries = False,
                image_shapes = None,
                name = None
            ))

        # fully connected network
        # [batch, length, hparams.ffn_size]
        x = residual_fn(x, transformer_ffn_layer(x, params))
        
    # convolution 
    convout = conv_op(x, params)
    
    channel_out = params['kernel'][-1]
    width_out = x.get_shape().as_list()[1]  - params['stride']


    # pooling over time
    convout = tf.reshape(convout, [-1, width_out, 1, channel_out])
    pooling = tf.nn.max_pool(convout, [1, width_out, 1, 1], [1, 1, 1, 1], 'VALID')
    pooling = tf.reshape(pooling, [-1, channel_out])
    
    ffn_out = ffn_op(pooling, params)
    
    softmax_out = tf.nn.softmax(ffn_out)
    
    predictions = tf.argmax(softmax_out, axis = -1)
    
    
    
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"sentiment": predictions})
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels = labels, logits = ffn_out)
    
    # accuracy as evaluation metric
    eval_metric_ops = { 
        'accuracy' : tf.metrics.accuracy(tf.argmax(labels, -1), predictions)
        }
    
    
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    optimizer = tf.train.AdamOptimizer()

    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
