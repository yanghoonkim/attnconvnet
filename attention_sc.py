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
    config = tf.contrib.learn.RunConfig(model_dir = FLAGS.model_dir, keep_checkpoint_max = 5)
    
    # load parameters
    model_params = getattr(params, FLAGS.params)().values()

    # define estimator
    nn = tf.estimator.Estimator(model_fn=model.attn_net, config = config, params=model_params)

    # define experiment
    exp_nn = tf.contrib.learn.Experiment(estimator = nn, 
            train_input_fn = train_input_fn, 
            eval_input_fn = test_input_fn,
            train_steps = FLAGS.steps,
            min_eval_frequency = None
            )

    # train and evaluate
    exp_nn.train_and_evaluate()

    #nn.train(input_fn=train_input_fn, steps=FLAGS.steps)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', type = str, default= '', help = 'path to the training input.')
    parser.add_argument('--train_target', type = str, default = '', help = 'path to the training target.')
    parser.add_argument('--test_input', type = str, default = '', help = 'path to the test input. ')
    parser.add_argument('--test_target', type = str, default = '', help = 'path to the test target.')
    parser.add_argument('--model_dir', type = str, help = 'path to save the model')
    parser.add_argument('--params', type = str, help = 'parameter setting')
    parser.add_argument('--steps', type = int, default = 200000, help = 'training step size')
    FLAGS = parser.parse_args()
    tf.app.run(main)
    
    
    # load preprocessed dataset
    #training_set = [np.load('../rcnn/part_train_max_20.npy'), np.load('../rcnn/part_train_max_20_target.npy')]
