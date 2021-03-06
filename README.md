# Cross Domain Normalization

### Introduction
This repository contains the Tensorflow implementation for the paper Cross Domain Normalization for Natural Language Object Retrieval. In this paper we address the task of finding an object in an image given an expression that refers to it. We focus on the model tendency to overfit due to the different update rates of the image and language models and show that by normalizing the statistics of the models outputs we can control the update rates and thereby reduce the overfit and significantly outperform state-of-the-art results. By just adding CDN, we increased the P@1 from 0.65 to 0.851 while stabilizing the training behavior and increasing the model's confidence level. Furthermore, CDN accelerates the training time significantly (with CDN it takes 2.5 minutes to outperform the best results, which we get after more than 3 hours without CDN).  We've tested our models on [Referit](http://tamaraberg.com/papers/referit.pdf) and used [SCRC](https://github.com/ronghanghu/natural-language-object-retrieval) and [GroundeR](https://github.com/kanchen-usc/GroundeR) among our baselines.

Note that the paper deals with a general mathematical phenomena that should be dealt with whenever few sub-models with widely different statistical distributions are combined. Thus, while we show the benefits of CDN for grounding textual phrases in images, it can be applied to other models that suffer for the same problem. Some of the code was taken from [here](https://github.com/andrewliao11/Natural-Language-Object-Retrieval-tensorflow).

### Prerequisites
<ul>
<li> Python 3.6
<li> Tensorflow 1.6
<li> Keras
<li> Opencv
<li> Scikit-learn
<li> Gensim
</ul>

## Running the Models
Before running the models and/or the notebooks, please follow these instructions:

1. Clone this git 
2. Download the ReferIt datasets from 
```
/datasets/ReferIt/ReferitData/download_ReferitData.sh
/datasets/ReferIt/ImageCLEF/download_data.sh
```
3. Pre-process the ReferIt dataset to generate metadata needed for training and evaluation by running ```python ./preprocess_dataset.py```
4. Cache the bbox features for train/test sets to disk (VGG16) by running: 
```
python ./exp-referit/train_cache_referit_local_features.py
python ./exp-referit/test_cache_referit_local_features.py
```
5. Pre-train words embeddings by running ```python ./exp-referit/w2v.py```
6. Build w2v dataset by running ```python ./exp-referit/cache_referit_datasets.py```

### Training the models

You can train and validate the paper's different models (using the manually annotated bounding boxes) via the notebooks in ```root/notebooks```, where root is the project's directory (the code for the models is in ```root/models```). This will allow you to examine the effect of different hyperparameters on the domains update rates, their statistics and the models performances. You can also run ```python ./train.py``` in order to train SG+CDND (see the paper for more details about SG+CDND). 

In order to train the models with proposals which were generated automatically, please do the following:
1. Download the spatial locations of the automatically generated boxes:
```
./data/download_edgebox_proposals.sh
```
2. Build the training and testing batches: 
```
python ./exp-referit/train_buildBatches_edgebox.py
python ./exp-referit/test_buildBatches_edgebox.py
```

3. Train the model by 
```python ./train_edgebox.py```
   : This will train SG+CDN. You can add the arguments ```-CDND True``` or ```-BiCDN True``` to run SGD+CDN or SG+BiCDN

To train the models we've used GPU (GeForce GTX 1080). The basic model (SG) took about 3.5 houres to converge while SG with CDN took about 2 hours. However, note that SG+CDN took about 2.5 minutes (one ephoc) to outperform the results we got without CDN. 

## Demo
Since we use Mask-RCNN in the demo, please do the following:

1. git clone https://github.com/matterport/Mask_RCNN.git to the project's root directory
2. Install pycocotools (which equires cython and a C compiler to install correctly):
    * Linux: ```git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI</li>```
    * Windows: ```pip install    git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI</li>```.
    Note that you'll need Visual C++ 2015 Build Tools for the windows version, which can be found in              http://landinghub.visualstudio.com/visual-cpp-build-tools

3. Download the mask_rcnn_coco.h5 file from  https://github.com/matterport/Mask_RCNN/releases to the project's root directory


Run the demo in ``` notbooks/demo.ipynb``` to get the object's segment, given the query. You can find images in the demo_images folder which contains mostly images from RefCLEF and were not used during the model's training. You can also run ```demo.py``` from the terminal, this will return the bounding box given an image and a query. 

## Performances

<b>Manually annotated bounding boxes:</b>

Model |Test P@1|Train P@1|Test Loss|Train Loss 
------|--------|---------|---------|-----------
RAND|0.294|-|-|-
CBoWG|0.62|0.706|1.944|2.03
SCRC|0.68|1|2.05|0.35
GroundeR|0.819|1|3.15|0.004
SG|0.66|0.99|3.64|0.47
SBN|0.831| 0.95|1.245|0.519
SG+CDN|0.845|0.948|1.11|0.475
SGD+CDN|0.851|0.93|1.04|0.6
SG+BiCDN|<b>0.86</b>|<b>0.96</b>|<b>1.04</b>|<b>0.44</b>

<b>Automatically generate bounding boxes:</b>

Model |Test P@1
------|--------
SCRC|17.93
GroundeR|26.93
MCB|28.91
Comprehension|31.85
SG|22.40
SG+CDN|33.60
SGD+CDN|34.00
SG+BiCDN|<b>34.40</b>
P@1 upperbound|59.38



<br><br>
The following table shows the results for SG with no BN (Batch Normalization), BN over the language model, image model, both and with scaled BN (adding BN layers over both language and image models and scaling the BN outputs). For these results and the grpah below we've used manually annotated bounding boxes

BN |Test P@1|Train P@1|Test Loss|Train Loss 
------|--------|---------|---------|-----------
None|0.65|0.99|3.8|0.11
Image|0.726|0.994|3.2|0.1
Language|0.8|1|3.03|0.002
Both|0.819|1|3.15|0.004
scaled|0.831|0.95|1.245|0.52

<br><br>
The graphs below show the dramatic effect of CDN in terms of acuuracy, regularization and training speed.

<p align="center">
   <b>SG</b><br>
  <img src="https://raw.githubusercontent.com/aSheffer/Cross-Domain-Normalization-for-Natural-Language-Object-Retrieval/master/graphs/simpleRes.png"><br><br>
    <b>SG with BN layers over the image and language models</b><br>
  <img src="https://raw.githubusercontent.com/aSheffer/Cross-Domain-Normalization-for-Natural-Language-Object-Retrieval/master/graphs/BNRes.png"><br><br>
    <b>SG with CDN</b><br>
  <img src="https://github.com/aSheffer/Cross-Domain-Normalization-for-Natural-Language-Object-Retrieval/blob/master/graphs/cdnRes.png">
</p>
