import numpy as np
import json
import os
import sys
from datetime import datetime
base_dir = os.path.dirname(os.path.realpath(__file__))+'/../'
sys.path.append(base_dir)
import retriever
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
import skimage
import cv2
import string 

save_edgbox_feat = base_dir+'data/train_edgbox_features/'
save_edgbox_batches = base_dir+'data/train_edgbox_batches/'
save_crop_toTrain = base_dir+'data/crop_toTrain.json'
vocab_file =  base_dir+'data/metadata/w2v_vocab.json'

proposal_dir = base_dir+'data/referit_edgeboxes_top100/'
train_imlist_path = base_dir+'data/split/referit_trainval_imlist.txt'
imcrop_dict_path = base_dir+'data/metadata/referit_imcrop_dict.json'
query_path = base_dir+'data/metadata/referit_query_dict.json'
imcrop_bbox_dict_path = base_dir+'data/metadata/referit_imcrop_bbox_dict.json'
image_dir = base_dir+'datasets/ReferIt/ImageCLEF/images/'

VGG = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
model = Model(VGG.input, VGG.layers[-2].output)

batch_size = 50 
k = 100 # How many bboxes per image

# Name of image crope -> Crope description
with open(query_path, 'r') as f:
    q = json.loads(f.read())
    
# Name of image -> Name of image crops
with open(imcrop_dict_path, 'r') as f:
    imcrop_dict = json.loads(f.read())

with open(train_imlist_path, 'r') as f:
    # List os all test images names 
    train_imlist = f.read().splitlines()
    
# Name of image crope -> Spatial location
with open(imcrop_bbox_dict_path, 'r') as f:
    bbox_dict = json.loads(f.read())

# loading vocabulary 
with open(vocab_file, 'r') as f:
    vocab = json.loads(f.read())
vocab['<unk>'] = len(vocab)

train_imlist.remove('30869')
num_im = len(train_imlist)

def build_candidate_boxes():
	# Image name -> 100 bboxes locations ([xmin, ymin, xmax, ymax])
	candidate_boxes_dict = {imname: None for imname in train_imlist}
	print('\nLoading training proposals...')
	for n_im in range(num_im):
	    if n_im % 1000 == 0:
	        print('Loading candidate regions %d / %d' % (n_im, num_im))
	    imname = train_imlist[n_im]
	    proposal_file_name = imname + '.txt'
	    boxes = np.loadtxt(proposal_dir + proposal_file_name)
	    candidate_boxes_dict[imname] = boxes.astype(int).reshape((-1, 4))
	return candidate_boxes_dict


def write_boxes(candidate_boxes_dict):
	print('\nBuilding training bboxes...')

	# Crop_name -> Whether any of the bboxes candidates has IOU>=0.5 with the crop
	crop_toTrain = dict()
	crops_num, tg_count = 0, 0 # recall = tg_count/crops_num
	time = datetime.now()
	er = 0

	if not os.path.exists(save_edgbox_feat):
	    os.makedirs(save_edgbox_feat)

	for n_im in range(num_im):
	    if n_im%100==0:
	        print('Building training bbox %d / %d' % (n_im, num_im))
	        if n_im>0:
	            print('Recall = %0.2f'%(tg_count/crops_num))
	            time_tmp = datetime.now()
	            print('Time:', time_tmp-time, '\n')
	            time = time_tmp 
	            
	    imname = train_imlist[n_im]
	    
	    # check whther ground truth exist
	    imcrop_names = imcrop_dict[imname]
	    crops_num+=len(imcrop_names)
	    candidate_boxes = candidate_boxes_dict[imname]
	    
	    # Whether any of the bboxes candidates has IOU>=0.5 with 
	    # any of the ground truth bbox in the image
	    flag = False
	    
	    for imcrop in imcrop_names:
	        if np.any(retriever.compute_iou(np.array(candidate_boxes), np.array(bbox_dict[imcrop]))>=0.5):
	            tg_count += 1
	            crop_toTrain[imcrop]=True
	            flag=True
	        else:
	            crop_toTrain[imcrop]=False
	            
	    if flag:
	        im = skimage.io.imread(image_dir + imname + '.jpg')
	        if len(im.shape)!=3:
	            print(imname, 'shape is', im.shape)
	            er+=1
	        else:
	            imsize = np.array([im.shape[1], im.shape[0]])  # [width, height]
	            spats = retriever.compute_spatial_feat(candidate_boxes, imsize)
	            [x1, y1, x2, y2] = candidate_boxes[:, 0], candidate_boxes[:, 1], candidate_boxes[:, 2], candidate_boxes[:, 3]
	            bboxes = [im[y1[n]:y2[n]+1, x1[n]:x2[n]+1, :] for n in range(candidate_boxes.shape[0])]
	            vgg_outputs = []
	            for b in bboxes:
	                resized_image = cv2.resize(b, (224,224)) 
	                resized_image=resized_image.astype(np.float32, copy=False)
	                resized_image[:,:,0] -= 103.939
	                resized_image[:,:,1] -= 116.779
	                resized_image[:,:,2] -= 123.68
	                vgg_outputs.append(resized_image)  

	            bbox_embeddings = model.predict(np.array(vgg_outputs))
	            np.savez(save_edgbox_feat+imname, spats=spats, vgg=bbox_embeddings)

	with open(save_crop_toTrain, 'w') as f:
	    f.write(json.dumps(crop_toTrain))
	    
	print('\nRecall =', tg_count/crops_num)
	return crop_toTrain


def processQ(q):
    '''
    Tokenize each query and pad the list of queries s.t all will have the same sizes
    '''
    
    exclude = set(string.punctuation)
    newQ = []

    for i in range(len(q)):
        clean = ''.join([ch.lower() if ch not in exclude else ' ' for ch in q[i].strip()]) # clean punctuation
        txt = [w for w in clean.split()]
        newQ.append([vocab[w] for w in txt])
        
    return newQ

def write_batches(candidate_boxes_dict, crop_toTrain, bsize=50):
	print('\nBuilding training batches...')
	if not os.path.exists(save_edgbox_batches):
	    os.makedirs(save_edgbox_batches)


	# List of ground truth bbox to train
	imList_new = [f.split('.')[0] for f in train_imlist]
	GTlist = [crop for crop in list(crop_toTrain.keys()) if crop_toTrain[crop]==True and crop.split('_')[0] in imList_new] 
	np.random.shuffle(GTlist)
	
	trnset_tmp = []
	for gt in GTlist:
	    for phrase in q[gt]:
	        trnset_tmp.append([phrase, gt])
	np.random.shuffle(trnset_tmp)

	trn_len =  len(trnset_tmp)
	attn_idx = [[1 for _ in range(100)] for _ in range(trn_len)]
	trn_gt = [item[1] for item in trnset_tmp]
	q_tmp = [item[0] for item in trnset_tmp]
	trn_q = processQ(q_tmp)
	images, vgg, spats, labels = [], [], [], []
	b = 0

	for k, crop in enumerate(trn_gt):
	    if k%1000==0:
	        print('Processing crop %d / %d'%(k, trn_len))
	    if k>0 and k%bsize==0:
	        start = b*bsize
	        end = (b+1)*bsize
	        trnset = [item for item in zip(*[attn_idx[start:end], 
	                                         trn_q[start:end], 
	                                         vgg[start:end], 
	                                         spats[start:end], 
	                                         labels[start:end]])]
	        
	        np.save(open(save_edgbox_batches+str(b)+'.bin', 'wb'), trnset)
	        b+=1
	        
	    imname = crop.split('_')[0]
	    candidate_boxes = candidate_boxes_dict[imname]
	    im = np.load(save_edgbox_feat+imname+'.npz')
	    
	    iou_list = retriever.compute_iou(np.array(candidate_boxes), np.array(bbox_dict[crop]))
	    gt_idx = np.argwhere(iou_list>=0.5)
	    assert len(gt_idx)>0, 'crop %d has no ground truth bbox'%(k)
	    
	    iou_sum = sum([iou_list[i[0]] for i in gt_idx])
	    labels.append([[i[0], iou_list[i[0]]/iou_sum] for i in gt_idx])

	    pedLengs = 100-im['vgg'].shape[0]
	    if pedLengs>0:
	        im_tmp = np.concatenate([im['vgg'], np.zeros((pedLengs, im['vgg'].shape[1]))], 0)
	        spats_tmp = np.concatenate([im['spats'], np.zeros((pedLengs, im['spats'].shape[1]))], 0)
	        attn_idx[k] = [1 for _ in range(im['spats'].shape[0])]+[0 for _ in range(pedLengs)]
	        vgg.append(im_tmp)
	        spats.append(spats_tmp)
	    else:
	        vgg.append(im['vgg'])
	        spats.append(im['spats'])



def main():
	candidate_boxes_dict = build_candidate_boxes()
	crop_toTrain = write_boxes(candidate_boxes_dict)
	write_batches(candidate_boxes_dict, crop_toTrain)

if __name__ == '__main__':
	main()