#########################################################################
# File Name: run_attention.sh
# Author: ad26kt
# mail: ad26kt@gmail.com
# Created Time: Mon 09 Oct 2017 05:07:43 PM KST
#########################################################################
#!/bin/bash
MODE='train'

trec(){
	TRAIN_INPUT='data/trec/processed/trec_train.npy'
	TRAIN_TARGET='data/trec/processed/trec_train_label.npy'
	TEST_INPUT='data/trec/processed/trec_test.npy'
	TEST_TARGET='data/trec/processed/trec_test_label.npy'
	PARAMS=trec_params
}

sst5(){
	TRAIN_INPUT='data/sst/processed_git/sst5_train.npy'
	TRAIN_TARGET='data/sst/processed_git/sst5_train_label.npy'
	TEST_INPUT='data/sst/processed_git/sst5_test.npy'
	TEST_TARGET='data/sst/processed_git/sst5_test_label.npy'
	PARAMS=sst5_params
}

sem4(){
	TRAIN_INPUT='data/semeval/processed/voc_train.npy'
	TRAIN_TARGET='data/semeval/processed/voc_train_label.npy'
	TEST_INPUT='data/semeval/processed/voc_test.npy'
	TEST_TARGET='data/semeval/processed/voc_test_label.npy'
	PARAMS=sem4_params
}

sem5(){
	TRAIN_INPUT='data/semeval/processed/ec_train.npy'
	TRAIN_TARGET='data/semeval/processed/ec_train_label.npy'
	TEST_INPUT='data/semeval/processed/ec_test.npy'
	TEST_TARGET='data/semeval/processed/ec_test_label.npy'
	PARAMS=sem5_params
}


# Pass the first argument as the name of dataset
$1

TRAIN_STEPS=200000
NUM_EPOCHS=None
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
