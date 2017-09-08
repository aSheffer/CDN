# Where am i

This is an analysis of performances and overfitting of text grounding in images using deep learning.<br>
Some of the code are from [here](https://github.com/andrewliao11/Natural-Language-Object-Retrieval-tensorflow)

## The Task
Given an image and a query that refers to an object in the image, we want to find the bounding box of the object.<br>
We use [Referit dataset](http://tamaraberg.com/referitgame/) which contains a set of images and for each image a set of 
queries that refere to an object in that image. The data set contains the boundaries of the bounding box for each query.
We use Referit to build our on dataset where each data point contains a query and all the bounding boxes of its image.
The model needs to find the correct bounding box.

## The base
We use the supervised model discribed in [Grounding of Textual Phrases in Images by
Reconstruction](https://arxiv.org/pdf/1511.03745.pdf).<br> 
Here's an image from the paper which illustrates the model: 
![ill](./images/base_model.png)
