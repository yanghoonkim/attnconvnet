import sys
sys.path.append('submodule/')

import numpy as np
import tensorflow as tf

import common_attention as ca
import common_layers as cl

# Evaluation metric for multi-label situation
def accuracy4multilabel(labels, predictions):
    numerator = tf.reduce_sum(tf.cast(tf.multiply(predictions, labels), tf.float32), axis = -1)
    denominator = tf.cast(tf.reduce_sum(predictions + labels, axis = -1), tf.float32) - numerator + 0.000001
    accuracy = tf.divide(numerator, denominator)
    mean, op = tf.metrics.mean(accuracy)
    return mean, op


def attn_net(features, labels, mode, params):
    hidden_size = params['hidden_size']
    voca_size = params['voca_size']
    
    def residual_fn(x, y):
        return cl.layer_norm(x + tf.nn.dropout(
            y, 1.0 - params['residual_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1))
    
    def embed_op(inputs, params):
        if params['embedding'] == None:
            embedding = tf.get_variable('embedding', [params['voca_size'], params['hidden_size']], dtype = params['dtype'])
        else:
            glove = np.load(params['embedding'])
            embedding = tf.Variable(glove, trainable = params['embedding_trainable'], name = 'embedding', dtype = tf.float32)

        tf.summary.histogram(embedding.name + '/value', embedding)
        return tf.nn.embedding_lookup(embedding, inputs)

    def conv_op(embd_inp, params):
        fltr = tf.get_variable(
                'conv_fltr', 
                params['kernel'], 
                params['dtype'], 
                regularizer = tf.contrib.layers.l2_regularizer(1.0)
                )

        convout = tf.nn.conv1d(embd_inp, fltr, params['stride'], params['conv_pad'])
        return convout
    def multi_conv_op(embed_inp, params):
        x = embed_inp
        return convout

    def ffn_op(x, params):
        out = x
        if params['ffn_size'] == None:
            ffn_size = []
        else:
            ffn_size = params['ffn_size']
        for unit_size in ffn_size[:-1]:
            out = tf.layers.dense(
                    out, 
                    unit_size, 
                    activation = tf.tanh, 
                    use_bias = True, 
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
                    )
        return tf.layers.dense(
                out, 
                params['label_size'], 
                activation = None, 
                use_bias = True, 
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
                )

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
            dropout=params['relu_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0)

    inputs = features['x']
    if params['lexicon_effect'] is not None:
        lexicon = features['lexicon']
        lexicon = tf.cast(lexicon, tf.float32)
    
    # raw input to embedded input of shape [batch, length, hidden_size]
    embd_inp = embed_op(inputs, params)
    if params['hidden_size'] != embd_inp.get_shape().as_list()[-1]:
        x = tf.layers.dense(
                embd_inp, 
                params['hidden_size'], 
                activation = None, 
                use_bias = False, 
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
                )
    else:
        x = embd_inp

    # attention bias computation
    padding = ca.embedding_to_padding(x)
    self_attention_bias = ca.attention_bias_ignore_padding(padding)

    for layer in xrange(params['num_layers']):
        x = residual_fn(
        x,
        ca.multihead_attention(
                query_antecedent = x,
                memory_antecedent = None, 
                bias = self_attention_bias,
                total_key_depth = params['hidden_size'],
                total_value_depth = params['value_depth'],
                output_depth = params['hidden_size'],
                num_heads = params['num_heads'],
                dropout_rate = params['attn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0,
                summaries = False,
                image_shapes = None,
                name = None
            ))

        # fully connected network
        # [batch, length, hparams.ffn_size]
        x = residual_fn(x, transformer_ffn_layer(x, params))

    logits = list()   
    for i in range(params['multi_label']):
        with tf.variable_scope(None, default_name = 'class'):
            # convolution 
            convout = conv_op(x, params)
    
            channel_out = params['kernel'][-1]
            # width_out = (feature_length - windows_size)/stride + 1
            # (56 - 10) + 1 = 47 
            width_out = (x.get_shape().as_list()[1] - params['kernel'][0])/params['stride'] + 1


            # pooling over time
            convout = tf.reshape(convout, [-1, width_out, 1, channel_out])
            pooling = tf.nn.max_pool(convout, [1, width_out, 1, 1], [1,1,1,1], 'VALID')
            pooling = tf.reshape(pooling, [-1, channel_out])

            if params['lexicon_effect'] == 'nrc1':
                lexicon_partial = tf.stack([lexicon[:,i]], axis = -1)
                integrate_lexicon = tf.concat([pooling, lexicon_partial], axis = -1)
                ffn_out = ffn_op(integrate_lexicon, params)
            else:
                ffn_out = ffn_op(pooling, params)

            logits.append(ffn_out)

    # predictions, loss and eval_metric 
    if len(logits) == 1:
        # single label classification
        logits = logits[0]
        softmax_out = tf.nn.softmax(logits)
        predictions = tf.argmax(softmax_out, axis = -1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                    mode = mode,
                    predictions = {'sentiment' : predictions})
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, params['label_size'])

        loss_ce = tf.losses.softmax_cross_entropy(onehot_labels = labels, logits = logits)
        eval_metric_ops = {
                'accuracy' : tf.metrics.accuracy(tf.argmax(labels, -1), predictions = predictions),
                'pearson_all' : tf.contrib.metrics.streaming_pearson_correlation(softmax_out, tf.cast(labels, tf.float32)),
                'pearson_some' : tf.contrib.metrics.streaming_pearson_correlation(tf.cast(predictions, tf.float32), tf.cast(tf.argmax(labels, -1), tf.float32))
                }
    else: 
        # multi label classification
        logits = tf.concat(logits, axis = -1)
        if mode != tf.estimator.ModeKeys.TRAIN and type(params['lexicon_effect']) == float:
            logits = logits + params['lexicon_effect'] * lexicon
        #predictions = tf.cast(tf.round(tf.sigmoid(logits)), tf.int32)
        prob = tf.sigmoid(logits)
        prob = tf.Print(prob, [prob], 'This is prob')
        predictions = tf.to_int32(prob>0.5)
        # Provide an estimator spec for 'Modekeys.PREDICT'
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                    mode = mode,
                    predictions = {"probability" : prob, "sentiment" : predictions})
        labels = tf.cast(labels, tf.int32)
        eval_metric_ops = {
                'accuracy' : accuracy4multilabel(labels, predictions)
                }
        loss_ce = tf.losses.sigmoid_cross_entropy(labels, logits)

    # regularizaiton loss
    loss_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    reg_const = params['regularization']  # Choose an appropriate one.
    
    loss = loss_ce + reg_const * loss_reg

    tf.summary.scalar('loss_ce', loss_ce)
    tf.summary.scalar('loss_reg', loss_reg)
    tf.summary.scalar('loss', loss)
    
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    learning_rate = params['learning_rate']
    learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), 500, params['decay'], staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    grad_and_var = optimizer.compute_gradients(loss, tf.trainable_variables())
    
    # add histogram summary for gradient
    for grad, var in grad_and_var:
        tf.summary.histogram(var.name + '/gradient', grad)
    train_op = optimizer.apply_gradients(grad_and_var, global_step = tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
