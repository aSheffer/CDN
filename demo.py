import os
import numpy as np
import skimage.io
import tensorflow as tf
import sys

import cv2
import json
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
import retriever
from CDND import Model as cdnd

ROOT_DIR =  os.path.dirname(os.path.realpath(__file__))

# Mask RCNN imports
sys.path.insert(0, ROOT_DIR+'/Mask_RCNN')
import coco
import utils
import model as modellib
import visualize

# Root directory of the project
ROOT_DIR =  os.path.dirname(os.path.realpath(__file__))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


vocab_file =  os.path.join(ROOT_DIR, 'data/metadata/w2v_vocab.json')
embed_path =  os.path.join(ROOT_DIR, 'data/metadata/w2v.bin')
params_dir = os.path.join(ROOT_DIR, 'data/training/models/All/unorder1_RL/EXP/CDNdrop')

# w2c words vectors
embed_vecs = np.load(open(embed_path, 'rb')).astype(np.float32)
# vocabulary
with open(vocab_file, 'r') as f:
    vocab = json.loads(f.read())
vocab['<unk>'] = len(vocab)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Get image meta data
def imgMeta(img_path, model):
    filename = img_path
    image = skimage.io.imread(filename)
    print(image.shape)
    meta = model.detect([image], verbose=1)
    return image, meta[0] 

# Get bboxes embeddings
def bboxVec(image, meta, vgg_model):
    bboxes = []
    spat = []
    for b in meta['rois']:
        bTrans = np.array([b[1], b[0], b[3], b[2]])
        imsize = np.array([image.shape[1], image.shape[0]])
        spat.append(retriever.compute_spatial_feat(bTrans, imsize))
        
        
        RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = RGB_im[b[0]:b[2],b[1]:b[3],:]
        resized_image = cv2.resize(crop, (224,224)) 
        resized_image=resized_image.astype(np.float32, copy=False)
        resized_image[:,:,0] -= 103.939
        resized_image[:,:,1] -= 116.779
        resized_image[:,:,2] -= 123.68
        v = vgg_model.predict(np.array([resized_image]))
        bboxes.append(v)
    
    return np.concatenate([bboxes, spat], 2)



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def main():
    config = InferenceConfig()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    VGG = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    vgg_model = Model(VGG.input, VGG.layers[-2].output)
    flag=True

    while flag:
        print()
        image_path = input("Please enter image path:")
        image_save = input("Please enter path to save bounding box:")
        phrase = input("Please enter your phrase:")
        print()
        # print('image_save:', image_save)
        image, meta = imgMeta(image_path, model)
        bboxes = bboxVec(image, meta, vgg_model)
        Qidx = [vocab[w] for w in phrase.split(' ')]

        images=bboxes[:,:,:4096] 
        spat=bboxes[:,:,4096:]
        tf.reset_default_graph()
        m = cdnd(batch_size=1)
        scores=m.predict(queries=[Qidx], img=images, bboxes=spat, attn_idx=[[1 for _ in range(images.shape[0])]])
        label = np.argmax(scores)
        b = meta['rois'][label]
        crop = image[b[0]-1:b[2]-1,b[1]-1:b[3]-1,:]

        cv2.imwrite( image_save, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        flag = input("Do you want ot continue? ([y/n]):")=='y'

if __name__ == "__main__":
    main()