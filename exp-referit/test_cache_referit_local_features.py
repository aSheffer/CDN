from __future__ import print_function, division

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import io

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model

import json
import tensorflow as tf
import skimage
import skimage.io

import h5py
import cv2

def main():
	VGG = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
	model = Model(VGG.input, VGG.layers[-2].output)

	image_dir = './data/resized_imcrop/'
	cached_context_features_dir = './data/test_referit_local_features/'
	imlist = io.load_str_list('./data/training/test_imcrop_list.txt')
	num_im = len(imlist)
	print("number of testing crops:", num_im)


	for i in range(len(imlist)):
		if imlist[i][-1]!='g':
		    imlist[i] = imlist[i]+'g'

	im2vec = dict()
	for j, im in enumerate(imlist):
	    if j%2000==0:
	        print('processing img number %d/%d'%(j, len(imlist)))
	    img = cv2.imread(image_dir+ im).astype(np.float32)
	    img[:,:,0] -= 103.939
	    img[:,:,1] -= 116.779
	    img[:,:,2] -= 123.68
	    im2vec[im] = model.predict(np.array([img]))


	if not os.path.isdir(cached_context_features_dir):
	    os.mkdir(cached_context_features_dir)
	for i, im in enumerate(imlist):
	    if i%200 == 0:
	        print('saving contextual features %d / %d' % (i, len(imlist)))
	    save_path = cached_context_features_dir + im.split('.')[0] + '_fc7.npy'
	    np.save(save_path, im2vec[im])


if __name__ == '__main__':
    main()