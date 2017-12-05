from __future__ import print_function, division
import sys
import os
import numpy as np
import string
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import io
import retriever


trn_imlist_file = './data/split/referit_trainval_imlist.txt'
tst_imlist_file = './data/split/referit_test_imlist.txt'
image_dir = './datasets/ReferIt/ImageCLEF/images/images/'
resized_imcrop_dir = './data/resized_imcrop/'
trn_cache_local_features_dir = './data/trainval_referit_local_features/'
tst_cache_local_features_dir = './data/test_referit_local_features/'
imcrop_dict_file = './data/metadata/referit_imcrop_dict.json'
imcrop_bbox_dict_file = './data/metadata/referit_imcrop_bbox_dict.json'
imsize_dict_file = './data/metadata/referit_imsize_dict.json'
query_file = './data/metadata/referit_query_dict.json'
trn_data_save = './data/training/w2v_train_data_new.bin'
tst_data_save = './data/training/w2v_test_data_new.bin'
vocab_path = './data/metadata/w2v_vocab.json'

trn_imset = set(io.load_str_list(trn_imlist_file)) # list of training images names
tst_imset = set(io.load_str_list(tst_imlist_file)) # list of training images names
query_dict = io.load_json(query_file) # imageName_boxName--> list of imageName_boxName discription
imsize_dict = io.load_json(imsize_dict_file) # imsize_dict[image name]-->[image width, image height]
imcrop_bbox_dict = io.load_json(imcrop_bbox_dict_file) # imageName_boxName --> [x_min, y_min, x_max, y_max] (box bounding cordinates)
imcrop_dict = io.load_json(imcrop_dict_file) # imcrop_dict[imageName] --> list(imageNames_boxName)

def main():
	# cleaning punctuation and build vocabulary 
	exclude, words = set(string.punctuation), []
	for k in query_dict.keys():
	    tmp=[]
	    for q in query_dict[k]:
	        clean = ''.join([ch.lower() if ch not in exclude else ' ' for ch in q.strip()]) # clean punctuation
	        words += [word for word in clean.split()]
	        tmp.append(clean.replace('  ', ' '))
	    query_dict[k] = tmp   

	# reading w2c vocab
	with open(vocab_path, 'r') as f:
		vocab = json.loads(f.read())

	# Train set 
	#list of [idx_q, (bbox_vec, true_bbox_feat), (bbox_vec, false_bbox_feat),... 
	#                          ...,(bbox_vec, false_bbox_feat), (index of ground truth)]
	train_pairs = [] 
	imcrop_spatial_dict = {imcrop:'' for imcrop in query_dict.keys()}

	for imcrop_name, des in query_dict.items():
	    imname = imcrop_name.split('_', 1)[0]
	    if imname not in trn_imset:
	        continue
	    imsize = np.array(imsize_dict[imname])
	    true_bbox = np.array(imcrop_bbox_dict[imcrop_name])
	    if imcrop_spatial_dict[imcrop_name]=='':
	        imcrop_spatial_dict[imcrop_name] = [np.load(trn_cache_local_features_dir + imcrop_name +  '_fc7.npy'), 
	                                            retriever.compute_spatial_feat(true_bbox, imsize)] 
	        
	    tg = (imcrop_spatial_dict[imcrop_name][0], imcrop_spatial_dict[imcrop_name][1])
	    for q in des:
	        clean = ''.join([ch.lower() if ch not in exclude else ' ' for ch in q.strip()]) # clean punctuation
	        idx_q = [vocab[w] for w in clean.strip().split()] 
	        train_tuple_tmp = []
	        for bname in imcrop_dict[imname]:
	            if bname!=imcrop_name:
	                bbox = np.array(imcrop_bbox_dict[bname])
	                if imcrop_spatial_dict[bname]=='':
	                    imcrop_spatial_dict[bname] = [np.load(trn_cache_local_features_dir + bname +  '_fc7.npy'), 
	                                                  retriever.compute_spatial_feat(bbox, imsize)]
	                train_tuple_tmp.append((imcrop_spatial_dict[bname][0], imcrop_spatial_dict[bname][1]))
	        if len(train_tuple_tmp)>0:
	            np.random.shuffle(train_tuple_tmp)
	            np.random.shuffle(train_tuple_tmp)
	            tg_idx = np.random.choice(range(len(train_tuple_tmp)+1))
	            train_tuple_tmp.insert(tg_idx,tg)
	            train_tuple_tmp.insert(0, idx_q) 
	            train_tuple_tmp.append(tg_idx)
	            train_pairs.append(train_tuple_tmp)

	np.random.shuffle(train_pairs)
	np.random.shuffle(train_pairs)
	np.save(open(trn_data_save, 'wb'), train_pairs)

	# Test set
	#list of [idx_q, (bbox_vec, true_bbox_feat), (bbox_vec, false_bbox_feat),... 
	#                          ...,(bbox_vec, false_bbox_feat), (index of ground truth)]
	tst_pairs = [] 
	imcrop_spatial_dict = {imcrop:'' for imcrop in query_dict.keys()}

	for imcrop_name, des in query_dict.items():
	    imname = imcrop_name.split('_', 1)[0]
	    if imname not in tst_imset:
	        continue
	    imsize = np.array(imsize_dict[imname])
	    true_bbox = np.array(imcrop_bbox_dict[imcrop_name])
	    if imcrop_spatial_dict[imcrop_name]=='':
	        imcrop_spatial_dict[imcrop_name] = [np.load(tst_cache_local_features_dir + imcrop_name + '_fc7.npy'), 
	                                            retriever.compute_spatial_feat(true_bbox, imsize)] 
	        
	    tg = (imcrop_spatial_dict[imcrop_name][0], imcrop_spatial_dict[imcrop_name][1])
	    for q in des:
	        clean = ''.join([ch.lower() if ch not in exclude else ' ' for ch in q.strip()]) # clean punctuation
	        idx_q = [vocab[w] for w in clean.strip().split()] 
	        tst_tuple_tmp = []
	        for bname in imcrop_dict[imname]:
	            if bname!=imcrop_name:
	                bbox = np.array(imcrop_bbox_dict[bname])
	                if imcrop_spatial_dict[bname]=='':
	                    imcrop_spatial_dict[bname] = [np.load(tst_cache_local_features_dir + bname + '_fc7.npy'), 
	                                                  retriever.compute_spatial_feat(bbox, imsize)]
	                tst_tuple_tmp.append((imcrop_spatial_dict[bname][0], imcrop_spatial_dict[bname][1]))
	                
	        if(len(tst_tuple_tmp)>0):
	            np.random.shuffle(tst_tuple_tmp)
	            np.random.shuffle(tst_tuple_tmp)
	            tg_idx = np.random.choice(range(len(tst_tuple_tmp)+1))
	            tst_tuple_tmp.insert(tg_idx,tg)
	            tst_tuple_tmp.insert(0, idx_q)
	            tst_tuple_tmp.append(tg_idx)
	            tst_pairs.append(tst_tuple_tmp)

	np.random.shuffle(tst_pairs)
	np.random.shuffle(tst_pairs)
	np.save(open(tst_data_save, 'wb'), tst_pairs)


if __name__ == '__main__':
    main()