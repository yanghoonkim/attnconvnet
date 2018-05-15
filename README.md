# AttnConvnet at SemEval-2018 Task 1 : Attention-based Convolutional Neural Networks for Multi-label Emotion Classification

Tensorflow implementation of [AttnConvnet](https://arxiv.org/pdf/1804.00831.pdf)

Warning!! **Dirty version source code**

The source code will be beautified and restructured ASAP

1. **Model**

	- Embedding
		- Pre-trained Glove embeddings
		- random initialized embeddings
	
	- Multi-head Dot-product Attention
		- using partial code from google's [Transformer](https://github.com/tensorflow/tensor2tensor)
	
	- 1-layer Convolutional Neural Network


2. **Dataset**
	
	- SemEval-2018 Task 1 : Affect in Tweets
		- [Subtask 5 : Detecting Emotions (multi-label classification)](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets)
			- [train-eng](http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-train.zip)
			- [dev-eng](http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-dev.zip)
			- [test-eng](http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-E-c-En-test.zip)
			
	- Lexicon
		- [Word-Emotion and Word-Sentiment Association lexicons](http://saifmohammad.com/WebPages/lexicons.html)
			- [NRC Word-Emotion Association Lexicon](http://saifmohammad.com/WebPages/AccessResource.htm)



## Requirements

- python 2.7
- numpy
- pandas
- Tensorflow 1.4
- nltk

## Usage

1. Download data to `data/`(In the root directory)

```
mkdir data
cd data
mkdir processed
wget http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-train.zip
wget http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-dev.zip
wget http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-E-c-En-test.zip

# unzip
unzip 2018-E-c-En-train.zip
unzip 2018-E-c-En-dev.zip
unzip 2018-E-c-En-test.zip
```

2. Emoji-to-meaning preprocessing(To be updated)

3. Get pre-trained embedding(Optional)

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```
4. Process data

```
cd ..
python process_data.py
python process_embedding.py # (optional) get pre-trained embedding, it will take a couple of minutes
```

5. Get & process nrc Lexicon file(Optional)
	- Put the file `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` inside `data/`
```
python process_nrc.py # it will take a couple of minutes
```

6. Train & Test model
```
# Train
bash run.sh [parameter set] train
# Example : bash run.sh basic_params train

# Text
bash run.sh [parameter set] pred
# Example : bash run.sh basic_params pred
```

## Use pre-trained embedding

Open `params.py` and change `embedding = None` to the path of pre-trained embedding file

Example : `embedding = 'data/processed/glove_embedding.npy'` 

## Use lexicon

Open `params.py` and change `lexicon_effect = None` to other values

- AC + nrc1 in [paper](https://arxiv.org/pdf/1804.00831.pdf)
	- `lexicon_effect = 'nrc1'`
- AC + nrc2 in [paper](https://arxiv.org/pdf/1804.00831.pdf)
	- `lexicon_effect = 0.4`

