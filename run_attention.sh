#########################################################################
# File Name: run_attention.sh
# Author: ad26kt
# mail: ad26kt@gmail.com
# Created Time: Mon 09 Oct 2017 05:07:43 PM KST
#########################################################################
#!/bin/bash

TRAIN_INPUT='../rcnn/trec_train_max_20.npy'
TRAIN_TARGET='../rcnn/trec_train_max_20_target.npy'
TEST_INPUT='../rcnn/trec_test_max_20.npy'
TEST_TARGET='../rcnn/trec_test_max_20_target.npy'

TRAIN_STEPS=200000

PARAMS=basic_params
MODEL_DIR=~/attentionnet/store_model/$PARAMS

python main.py \
	--train_input=$TRAIN_INPUT \
	--train_target=$TRAIN_TARGET \
	--test_input=$TEST_INPUT \
	--test_target=$TEST_TARGET \
	--model_dir=$MODEL_DIR \
	--params=$PARAMS \
	--steps=$TRAIN_STEPS
