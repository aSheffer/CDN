# Where am I?

In this work we analys the performances of text grounding in images using deep learning.<br>
Some of the code are from [here](https://github.com/andrewliao11/Natural-Language-Object-Retrieval-tensorflow)

## The Task

Given a query that refers to an object in an image, we want to find the bounding box of that object.<br>
We use [Referit dataset](http://tamaraberg.com/referitgame/) in which each data point contains an image, a textual reference to an object in that image and the spatial boundaries of the object's bounding box.<br>
Most images in the Referit dataset have more than one referential objects, we use that fact to build our on dataset in the following way: Each image with n (n>1) referential objects is divided to n sub-images, one per referential object, with this each of our data point contains
<ul>
<il>The query</il>
<il>A list of size n. The i'th item in the list contains [a sub-image, its bouding box bounderies in the image]</il>
<li>The number of the list's item to which the query refers to.</il>
</ul>

a query and all  all images with more the one  Each image in the Referit dataset has multiple  
Using Referit dataet, we build our on dataset where each data point contains a query and all the bounding boxes of its image.
The model needs to find the correct bounding box.

# The base
We use the supervised model discribed in [Grounding of Textual Phrases in Images by
Reconstruction](https://arxiv.org/pdf/1511.03745.pdf).<br> 
Here's an image from the paper which illustrates the model:<br>
![ill](./images/base_model.png)
<ul>
<il>We use RNN to embed the query (LSTM)</il>
<il>We use CNN (VGG16) to embed each of bboxes</il>
<il>We use attention mechanism to get the score of each bbox, given the query, and the highest scored bbox is our winer! </il> 
</ul>

## Base on Steroids 

We also try the add the base model attention on the word level, that is, each time step attend on all the bboxes and by that we ground the words and not only tahe entire query. <br>
We also try to use bi-directional rnn since the relation between an object deccribed in time step t to other objects might be written in time steps smaller and/or higher than t. 

# Regularization

We examine state-of-the-art regularization techniques as [batch normalization](https://arxiv.org/abs/1502.03167), [dropout](https://arxiv.org/pdf/1207.0580.pdf), normal noise, L2 and gradient cliping. In additioin we porpuse a new technick that might shed some light on the model's performances useing reinforcment in an adversarial setting. 

## Player A

We add to our model another player, we call it A, while the base model is called B.<br>
While B's rnn process the query, at each time step B's output is the input of A's rnn (again, LSTM). A's rnn output is then projectes to two values, one for each of the two possible actions:<br>
1. Change the query's word (in time step t) to the word 'unk' and run B again with 'unk' as the input word instead of the original word - this will now be the B's output for time step t and get negative reward.
2. Do nothing and get zero reward. 
<p>A  will take the action with the higer value and by that potentially making B's loss higher.</p>

### In more details

Before we start A, we start by running B on the original query (with no edits). This will give us B's loss and outputs (on the un-edited query). If A decides to edit a word it will get a reward equals to this loss divided by the number of words and multiplied by some negative number, this means that editing a word becomes less attractive as the un-edited loss gets bigger and/or the number of words gets smaller.<br>

Then we run B and A together. At the end of this run we get B's loss on the edited query and use it as the final reward. Using bellman equation, we calculate the value of each time step and the loss in time step t is the MSE between it and the A's value for the chosen action.

### A inputs

At each time step A gets - B's rnn output for that time step, the reward dor editing a word, B's loss on the un-edited query.
In addition we add attention over B's outputs on the un-edited query.<br>

### NOTE

A does not 'see' any of the words nor any of the images. This means that A can't learn a good languish model nor does it know anything on B's task (since it doesn't 'see' the images), therefore A can't learn how to attentionally interfere B's learning a good languish model. Actually, the only thing A can learn is to recognize overfitting patterns by looking at B's features, so A might give us some insight of how B's working considering the huge variance in both vision and languish.

# Baseline

As a baseline we used a pre-trained w2v model to initialize the word embadding. We then train a model that calculate cosine similarity between the average of a query's words vectors and a progection of the bboxes vectors. 

# Runnging The Models

You can find the models codes and results in the notebooks folder.

## Requirement

You'll need Opencv, Keras, Tensorflow and python 3+  

## Preprocess

1. Download the ReferIt dataset: ./datasets/ReferIt/ReferitData/download_ReferitData.sh and ./datasets/ReferIt/ImageCLEF/download_data.sh
2. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: python ./preprocess_dataset.py
3. Cache the bbox features for train/test sets to disk: python ./exp-referit/train_cache_referit_local_features.py and python ./exp-referit/test_cache_referit_local_features.py
4. Build dataset: python ./exp-referit/cache_referit_training_batches.py
5. Train word2vec: python ./exp-referit/w2v.py
6. Build w2v baseline dataset: python ./exp-referit/w2v_train_cache_referit_local_featurs.py

# Notebooks

All the codes and the results can be founds in the notebooks folder.
