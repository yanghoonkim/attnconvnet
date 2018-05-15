# AttnConvnet at SemEval-2018 Task 1 : Attention-based Convolutional Neural Networks for Multi-label Emotion Classification

Tensorflow implementation of AttnConvnet

1. **Model**

	- Embedding
		- Pre-trained Glove embeddings
		- random initialized embeddings
	
	- Multi-head Dot-product Attention
		- using partial code from google's [Transformer](https://github.com/tensorflow/tensor2tensor)
	
	- 1-layer Convolutional Neural Network


2. **Dataset**
	
	- SemEval-2018 Task 1 : Affect in Tweets
		- Subtask 5 : Detecting Emotions (multi-label classification)

	- Lexicon
