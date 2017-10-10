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

VALID_INPUT=None
VALID_TARGET=None

TRAIN_STEPS=200000

PARAMS=basic_params

python attention_sc.py \
	--train_input=$TRAIN_INPUT \
	--train_target=$TRAIN_TARGET \
	--test_input=$TEST_INPUT \
	--test_target=$TEST_TARGET \
	--params=$PARAMS \
	--steps=$TRAIN_STEPS
