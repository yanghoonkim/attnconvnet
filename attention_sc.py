import argparse
import numpy as np
import tensorflow as tf

import params
import model

FLAGS = None


def main(unused):
    
    # Enable logging for tf.estimator
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # load preprocessed dataset
    training_set = [np.load(FLAGS.train_input), np.load(FLAGS.train_target)]
    test_set = [np.load(FLAGS.test_input), np.load(FLAGS.test_target)]
    
    # training input function for estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set[0])},
        y=np.array(training_set[1]),
        num_epochs=None,
        shuffle=True)
    
    # test input function for estimator
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set[0])},
        y=np.array(test_set[1]),
        num_epochs=1,
        shuffle=False)
    
    # config
    config = tf.contrib.learn.RunConfig(model_dir = './', keep_checkpoint_max = 5)
    
    # load parameters
    model_params = getattr(params, FLAGS.params)().values()

    # define estimator
    nn = tf.estimator.Estimator(model_fn=model.attn_net, config = config, params=model_params)
    
    nn.train(input_fn=train_input_fn, steps=FLAGS.steps)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', type = str, default= '', help = 'Path to the training input.')
    parser.add_argument('--train_target', type = str, default = '', help = 'Path to the training target.')
    parser.add_argument('--test_input', type = str, default = '', help = 'Path to the test input.') 
    parser.add_argument('--test_target', type = str, default = '', help = 'Path to the test target.')
    parser.add_argument('--params', type = str, help = 'Parameter setting')
    parser.add_argument('--steps', type = int, default = 200000, help = 'Training step size')
    FLAGS = parser.parse_args()
    tf.app.run(main)
    
    
    # load preprocessed dataset
    #training_set = [np.load('../rcnn/part_train_max_20.npy'), np.load('../rcnn/part_train_max_20_target.npy')]
