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

trec(){
	TRAIN_INPUT='data/trec/processed/trec_train.npy'
	TRAIN_TARGET='data/trec/processed/trec_train_label.npy'
	DEV_INPUT='data/trec/processed/trec_test.npy'
	DEV_TARGET='data/trec/processed/trec_test_label.npy'
	PARAMS=trec_params
}

sst5(){
	TRAIN_INPUT='data/sst/processed_git/sst5_train.npy'
	TRAIN_TARGET='data/sst/processed_git/sst5_train_label.npy'
	DEV_INPUT='data/sst/processed_git/sst5_test.npy'
	DEV_TARGET='data/sst/processed_git/sst5_test_label.npy'
	PARAMS=sst5_params
}

sem4(){
	TRAIN_INPUT='data/semeval/processed/voc_train1.npy'
	TRAIN_TARGET='data/semeval/processed/voc_train1_label.npy'
	DEV_INPUT='data/semeval/processed/voc_dev.npy'
	DEV_TARGET='data/semeval/processed/voc_dev_label.npy'
	TEST_INPUT='data/semeval/processed/voc_dev.npy'
	TEST_ORIGIN='data/semeval/2018-Valence-oc-En-dev.txt'
	PRED_DIR='result/sem4/V-oc_en_pred.txt'
	PARAMS=sem4_params
}

sem5(){
	TRAIN_INPUT='data/semeval/processed/ec_train_all.npy'
	TRAIN_TARGET='data/semeval/processed/ec_train_label_all.npy'
	DEV_INPUT='data/semeval/processed/ec_dev_all.npy'
	DEV_TARGET='data/semeval/processed/ec_dev_label_all.npy'
	TEST_INPUT='data/semeval/processed/ec_mass.npy'
	#TEST_ORIGIN='data/semeval/2018-E-c-En-dev.txt'
	#TEST_ORIGIN='data/semeval/2018-E-c-EN-test.txt'
	TEST_ORIGIN='data/semeval/SemEval2018-AIT-DISC-tweets2_max40.txt'
	#PRED_DIR='result/sem5/E-C_en_pred.txt'
	PRED_DIR='result/sem5/mass39k.txt'
	PARAMS=sem5_params_ignore_bias
}

sem5c1(){
	TRAIN_INPUT='data/semeval/processed/ec_mass.npy'
	TRAIN_TARGET='data/semeval/processed/ec_mass_label.npy'
	DEV_INPUT='data/semeval/processed/ec_dev_all.npy'
	DEV_TARGET='data/semeval/processed/ec_dev_label_all.npy'
	TEST_INPUT='data/semeval/processed/ec_mass.npy'
	#TEST_ORIGIN='data/semeval/2018-E-c-En-dev.txt'
	#TEST_ORIGIN='data/semeval/2018-E-c-EN-test.txt'
	TEST_ORIGIN='data/semeval/SemEval2018-AIT-DISC-tweets2_max40.txt'
	#PRED_DIR='result/sem5/E-C_en_pred.txt'
	PRED_DIR='result/sem5/mass39k.txt'
	PARAMS=sem5_only_unlabel
}

sem5emo(){
	TRAIN_INPUT='data/semeval/processed/ec_train_emo_unlabel.npy'
	TRAIN_TARGET='data/semeval/processed/ec_train_emo_label_unlabel.npy'
	DEV_INPUT='data/semeval/processed/ec_dev_emo_unlabel.npy'
	DEV_TARGET='data/semeval/processed/ec_dev_emo_label_unlabel.npy'
	TEST_INPUT='data/semeval/processed/ec_mass.npy'
	#TEST_ORIGIN='data/semeval/2018-E-c-En-dev.txt'
	#TEST_ORIGIN='data/semeval/2018-E-c-EN-test.txt'
	TEST_ORIGIN='data/semeval/SemEval2018-AIT-DISC-tweets2_max40.txt'
	#PRED_DIR='result/sem5/E-C_en_pred.txt'
	PRED_DIR='result/sem5/mass39k.txt'
	PARAMS=sem5_emo
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
	--model_dir=$MODEL_DIR \
	--pred_dir=$PRED_DIR \
	--test_origin=$TEST_ORIGIN \
	--params=$PARAMS \
	--steps=$TRAIN_STEPS\
	--num_epochs=$NUM_EPOCHS
