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
Before running the models and/or the notbookes, please follow these instructions:

1. Clone this git 
2. Download the ReferIt dataset in 
```/datasets/ReferIt/ReferitData/download_ReferitData.sh
/datasets/ReferIt/ImageCLEF/download_data.sh```
3. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: <b>python ./preprocess_dataset.py</b>
4. Cache the bbox features for train/test sets to disk (VGG16): <b>python ./exp-referit/train_cache_referit_local_features.py</b> and <b>python ./exp-referit/test_cache_referit_local_features.py</b>
5. Pre-train words embeddings: <b>python ./exp-referit/w2v.py</b>
6. Build w2v dataset: <b>python ./exp-referit/cache_referit_datasets.py</b>j

### Training the models

You can train and validate the the papaer's different models via the notebooks in the the notebooks folder. This will allow you to examine the effect of different hyper parameters on the domains update rates, their statistics and the models performances.
You can also run <b>'python ./train.py'</b> in order to train SG+CDND (see the paper for more details about SG+CDND)

## Demo
Since we use Mask-RCNN in the demo, please do the following:

1. git clone https://github.com/matterport/Mask_RCNN.git to the project root directory
2. Install pycocotools
3. Download the mask_rcnn_coco.h5 file from  https://github.com/matterport/Mask_RCNN/releases to the project root directory


Run ``` notbooks/demo.ipynb``` for demo examples to get the obgect's segment, given the query. You can find images at the demo_images folder which contains mostly images from RefCLEF and were not used during the model's trining. You can also run ```demo.py``` from the terminal, this will return the bounding box given an image and a query. 



