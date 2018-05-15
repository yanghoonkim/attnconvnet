#########################################################################
# File Name: run_attention.sh
# Author: ad26kt
# mail: ad26kt@gmail.com
# Created Time: Mon 09 Oct 2017 05:07:43 PM KST
#########################################################################
#!/bin/bash
train(){
	MODE='train'
}

eval(){
	MODE='eval'
}

pred(){
	MODE='pred'
}

basic_params(){
	TRAIN_INPUT='data/processed/ec_train.npy'
	TRAIN_TARGET='data/processed/ec_train_label.npy'
	DEV_INPUT='data/processed/ec_dev.npy'
	DEV_TARGET='data/processed/ec_dev_label.npy'
	TEST_INPUT='data/processed/ec_test.npy'
	LEXICON_TRAIN='data/processed/sentiment_train.npy'
	LEXICON_DEV='data/processed/sentiment_dev.npy'
	LEXICON_TEST='data/processed/sentiment_test.npy'
	TEST_ORIGIN='data/2018-E-c-En-test.txt'
	PRED_DIR='result/E-C_en_pred.txt'
	PROB_DIR='result/E-C_en_prob.txt'
	PARAMS=basic_params
}


# Pass the first argument as the name of dataset
# Pass the second argument as mode
$1
$2

TRAIN_STEPS=200000
NUM_EPOCHS=None
MODEL_DIR=~/work/attentionnet/store_model/$PARAMS

python main.py \
	--mode=$MODE \
	--train_data=$TRAIN_INPUT \
	--train_label=$TRAIN_TARGET \
	--eval_data=$DEV_INPUT \
	--eval_label=$DEV_TARGET \
	--test_data=$TEST_INPUT \
	--lexicon_train=$LEXICON_TRAIN \
	--lexicon_dev=$LEXICON_DEV \
	--lexicon_test=$LEXICON_TEST \
	--model_dir=$MODEL_DIR \
	--pred_dir=$PRED_DIR \
	--prob_dir=$PROB_DIR \
	--test_origin=$TEST_ORIGIN \
	--params=$PARAMS \
	--steps=$TRAIN_STEPS\
	--num_epochs=$NUM_EPOCHS
