This is an analysis of performances and overfitting of text grounding in images using deep learning.
Some of the code are from [Natural Language Object Retrieval in Tensorflow](https://github.com/andrewliao11/Natural-Language-Object-Retrieval-tensorflow)
<br><br><br>
# The Task
Given an image and a query that refers to an object in the image, we want to find the bounding box of the object.<br>
We use [Referit dataset](http://tamaraberg.com/referitgame/) which contains a set of images and for each image a set of 
queries that refere to an object in that image. The data set contains the boundaries of the bounding box for each query.
We use Referit to build our on dataset where each data point contains a query and all the bounding boxes of its image.
The model needs to find the correct bounding box.

# The base
We use the supervised model discribed in [Grounding of Textual Phrases in Images by
Reconstruction](https://arxiv.org/pdf/1511.03745.pdf).<br> 
Here's an image from the paper which illustrates the model: 
![alt text](https://github.com/aSheffer/GAB/tree/master/images/base_model.png)
