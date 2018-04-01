# Cross Domain Normalization

### Introduction

This repository contains the Tensorflow implementation for the paper Cross Domain Normalization for Natural Language Object Retrieval. In this paper we address the task of finding an object in an image given an expression that refers to it. We focus on the model tendency to overfit due to the different update rates of the image and language models and show that by normelizing the statistics of the models outputs we can contorl the update rates and thereby reduce the overfit and improve the model's performances. Some of the code are taken from [here](https://github.com/ronghanghu/natural-language-object-retrieval)

### Prerequisites
<ul>
<li> Python 3.6
<li> Tensorflow 1.6
<li> Keras
<li> Opencv
<li> Scikit-learn
<li> Gensim
</ul>

# The Grounder
The model is based on the supervised model, discribed in [Grounding of Textual Phrases in Images by
Reconstruction](https://arxiv.org/pdf/1511.03745.pdf).<br> 
The following image from the paper illustrates the model:<br>
![ill](./images/base_model.png)
<ul>
<li>We use RNN (LSTM) to embed the query</li>
<li>We use CNN (VGG16) to embed each of one the sub-images</li>
<li>We use attention mechanism to get the score of each bbox, given the query, and the highest scored bbox is our winer! </li> 
</ul>

## Grounder on Steroids 

We test the model with different hyperparameters and different regularization techniques such as [batch normalization](https://arxiv.org/abs/1502.03167), [dropout](https://arxiv.org/pdf/1207.0580.pdf), normal noise, L2 and gradient cliping. In additioin we porpuse a new regularization technique, that might shed some light on the model's performances, using reinforcment in an adversarial setting (see the next section).<br>
The problem's nature to overfit might lead us to pass on good models just because of their size, Therefore we also examine the model performence after replacing the RNN by a bi-directional RNN and adding a word level attention mechanism (i.e. at each time step we use attention over all the sub-images) there by adding a word level grounding. We test these models with and without reularization.

## Reinforcment  as a Regularization

To make things a bit harder for the grounder, we add another player - A. While the Grounder run its RNN, at time step t A can do one of two actions, depending on the Grounder's RNN t'th output:
<ol> 
<li>Edit the t'th word to 'unk'. This forces the Grounder to run the t'th time step again with the word 'unk', ignoring it's privious result. This action comes with some negative reward for A.</li>
<li> Do nothing. This has a zero reward.
</ol>
When the Grounder finishes to process the query, A gets a reward equals to the Grounder loss<br><br>
A follows a Q-earning mechanism in which it tries to predict the value for each <state, action> pair (where the state is the Grounder RNN's output) and chooses the action with the higher value. 

###  Implementation notes

In time step t we feed the query t'th word to the Grounder and its output to A's RNN cell. But at each iteration, before activating A, we feed the un-edited query to the Grounder, A uses an attention mechanism over these un-edited outputs. A will then decide on an action and the Grounder will act as explained above.<br><br>

Note that A does not 'see' any of the words nor any of the possible sub-images. This means that A can't learn a good languish model nor does it know anything on the Grounder's task (since it doesn't 'see' the 'labels'), therefore A can't learn how to intentionally interfere the Grounder from learning a good languish model. Actually, the only thing A can learn is to recognize overfitting patterns by looking at the Grounder outputs features, so A might give us some insight on how the Grounder works considering the huge variance in both vision and languish.

# Baseline

As a baseline we used a pre-trained w2v model to initialize the words embadding. We then build a model that calculates the cosine similarity between the average of a query's words vectors and the bboxes vectors projection. This model is trained to maximize the cosine similarity between the query and the true bbox while minimizing the cosine similarity between the query and the others bboxes.

# Runnging The Models

You can find the models codes and results in the notebooks folder.

## Requirement

You'll need Opencv, Keras, Tensorflow and python 3+  

## Preprocess new
1. Clone the git 

2. Download the ReferIt dataset in <b>./datasets/ReferIt/ReferitData/download_ReferitData.sh</b> and <b>./datasets/ReferIt/ImageCLEF/download_data.sh</b>

3. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: <b>python ./preprocess_dataset.py</b>

4. Cache the bbox features for train/test sets to disk (VGG16): <b>python ./exp-referit/train_cache_referit_local_features.py</b> and <b>python ./exp-referit/test_cache_referit_local_features.py</b>

4*. Build dataset: <b>python ./exp-referit/cache_referit_training_batches.py</b> \\Delete

5. Train word2vec: <b>python ./exp-referit/w2v.py</b>
6. Build w2v dataset: <b>python ./exp-referit/cache_referit_datasets.py</b>

7*. Build batches: <b>python ./exp-referit/build_batch.py\\Delete


## Preprocess

1. Download the ReferIt dataset: <b>./datasets/ReferIt/ReferitData/download_ReferitData.sh</b> and <b>./datasets/ReferIt/ImageCLEF/download_data.sh</b>
2. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: <b>python ./preprocess_dataset.py</b>
3. Cache the bbox features for train/test sets to disk (VGG16): <b>python ./exp-referit/train_cache_referit_local_features.py</b> and <b>python ./exp-referit/test_cache_referit_local_features.py</b>
4. Build dataset: <b>python ./exp-referit/cache_referit_training_batches.py</b>
5. Train word2vec: <b>python ./exp-referit/w2v.py</b>
6. Build w2v dataset: <b>python ./exp-referit/w2v_cache_referit_training_batches.py</b>
7. Build batches: <b>python ./exp-referit/build_batch.py
