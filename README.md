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

## Runnging The Models
Before running the model and/or the notbookes, please follow these instructions:

1. Clone this git 
2. Download the ReferIt dataset in <b>./datasets/ReferIt/ReferitData/download_ReferitData.sh</b> and <b>./datasets/ReferIt/ImageCLEF/download_data.sh</b>
3. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: <b>python ./preprocess_dataset.py</b>
4. Cache the bbox features for train/test sets to disk (VGG16): <b>python ./exp-referit/train_cache_referit_local_features.py</b> and <b>python ./exp-referit/test_cache_referit_local_features.py</b>
5. Train word2vec: <b>python ./exp-referit/w2v.py</b>
6. Build w2v dataset: <b>python ./exp-referit/cache_referit_datasets.py</b>
