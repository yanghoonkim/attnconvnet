#########################################################################
# File Name: run_attention.sh
# Author: ad26kt
# mail: ad26kt@gmail.com
# Created Time: Mon 09 Oct 2017 05:07:43 PM KST
#########################################################################
#!/bin/bash
MODE='train'

TRAIN_INPUT='data/sst/processed_git/sst5_train.npy'
TRAIN_TARGET='data/sst/processed_git/sst5_train_label.npy'
TEST_INPUT='data/sst/processed_git/sst5_test.npy'
TEST_TARGET='data/sst/processed_git/sst5_test_label.npy'

TRAIN_STEPS=200000
NUM_EPOCHS=None

PARAMS=test_params
MODEL_DIR=~/work/attentionnet/store_model/$PARAMS

python main.py \
	--mode=$MODE \
	--train_data=$TRAIN_INPUT \
	--train_label=$TRAIN_TARGET \
	--eval_data=$TEST_INPUT \
	--eval_label=$TEST_TARGET \
	--model_dir=$MODEL_DIR \
	--params=$PARAMS \
	--steps=$TRAIN_STEPS\
	--num_epochs=$NUM_EPOCHS
