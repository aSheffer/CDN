from __future__ import division, print_function

import os
import numpy as np
import scipy.io as sio
import skimage
import skimage.io
import skimage.transform
import warnings
from util import io
import json

def load_imcrop(imlist, mask_dir):
    '''
    returns:
        imcrop_dict: imcrop_dict[image name] --> list(image' boxes names)
        imcroplist: list of image' boxes names
    '''
    # imcrop_dict[image name] --> list(image' boxes names)
    # example: 25_1.jpg
    imcrop_dict = {im_name: [] for im_name in imlist}
    imcroplist = []
    masklist = os.listdir(mask_dir)
    for mask_name in masklist:
        imcrop_name = mask_name.split('.', 1)[0]
        imcroplist.append(imcrop_name)
        im_name = imcrop_name.split('_', 1)[0]
        imcrop_dict[im_name].append(imcrop_name)
    return imcroplist, imcrop_dict


def load_image_size(imlist, image_dir):
    '''
    returns dict: imsize_dict[image name]-->[image width, image height]
    '''
    num_im = len(imlist)
    imsize_dict = {}
    for n_im in range(num_im):
        if n_im % 200 == 0:
            print('processing image size %d / %d' % (n_im, num_im))
        im = skimage.io.imread(image_dir + imlist[n_im] + '.jpg')
        imsize_dict[imlist[n_im]] = [im.shape[1], im.shape[0]]  # [width, height]
    return imsize_dict


def load_referit_annotation(imcroplist, annotation_file):
    '''
    params:
        mcroplist: list of image' boxes names
        annotation_file: RealGames.text file (each row is: 'imageName_boxName.jpg~discription~x click~y click')
    
    returns:
        query_dict: imageName_boxName--> list of imageName_boxName discriptions
    '''
    
    print('loading ReferIt dataset annotations...')
    query_dict = {imcrop_name: [] for imcrop_name in imcroplist}
    with open(annotation_file) as f:
        # example: "8756_2.jpg~sunray at very top~.33919597989949750~.023411371237458192"
        raw_annotation = f.readlines()
    for s in raw_annotation:  
        
        # split: [imageName_boxName.jpg, discription, x~y]
        splits = s.strip().split('~', 2)   
        
        # imcrop_name: imageName_boxName
        imcrop_name = splits[0].split('.', 1)[0]
        
        description = splits[1]
        
        # construct imcrop_name - discription list dictionary
        # an image crop can have zero or mutiple annotations
        query_dict[imcrop_name].append(description)
    return query_dict


def load_and_resize_imcrop(mask_dir, image_dir, resized_imcrop_dir):
    '''
    Resizinig boxes to 224 x 224, saving them in resized_imcrop_dir
    and returns imcrop_bbox_dict: imageName_boxName --> [x_min, y_min, x_max, y_max] (box bounding cordinates)
        
    '''
    print('loading image crop bounding boxes...')
    imcrop_bbox_dict = {}
    masklist = os.listdir(mask_dir)
    if not os.path.isdir(resized_imcrop_dir):
        os.mkdir(resized_imcrop_dir)
    for n in range(len(masklist)):
        if n % 200 == 0:
            print('processing image crop %d / %d' % (n, len(masklist)))
        mask_name = masklist[n] # imageName_boxName.mat
        mask = sio.loadmat(mask_dir + mask_name)['segimg_t']
        
        # idx:indices of the elements that are non-zero in mask
        idx = np.nonzero(mask == 0) 
        x_min, x_max = np.min(idx[1]), np.max(idx[1])
        y_min, y_max = np.min(idx[0]), np.max(idx[0])
        bbox = [x_min, y_min, x_max, y_max]
        imcrop_name = mask_name.split('.', 1)[0] # imageName_boxName
        imcrop_bbox_dict[imcrop_name] = bbox

        # resize the image crops
        imname = imcrop_name.split('_', 1)[0] + '.jpg'
        image_path = image_dir + imname
        im = skimage.io.imread(image_path)
        # Gray scale to RGB
        if im.ndim == 2:
            im = np.tile(im[..., np.newaxis], (1, 1, 3))
        
        # RGBA to RGB
        im = im[:, :, :3]
        resized_im = skimage.transform.resize(im[y_min:y_max+1,
                                                 x_min:x_max+1, :], [224, 224])
        save_path = resized_imcrop_dir + imcrop_name + '.png'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(save_path, resized_im)
    return imcrop_bbox_dict

def main():
    image_dir = './datasets/ReferIt/ImageCLEF/images/'
    mask_dir = './datasets/ReferIt/ImageCLEF/mask/'
    annotation_file = './datasets/ReferIt/ReferitData/RealGames.txt'
    imlist_file = './data/split/referit_all_imlist.txt'
    metadata_dir = './data/metadata/'
    resized_imcrop_dir = './data/resized_imcrop/'

    imlist = io.load_str_list(imlist_file)
    imsize_dict = load_image_size(imlist, image_dir) # imsize_dict[image name]-->[image width, image height]
    imcroplist, imcrop_dict = load_imcrop(imlist, mask_dir) # imcrop_dict[image name] --> list(image' boxes names)
    query_dict = load_referit_annotation(imcroplist, annotation_file) # imageName_boxName--> list of imageName_boxName discription
    imcrop_bbox_dict = load_and_resize_imcrop(mask_dir, image_dir, resized_imcrop_dir) # imageName_boxName --> [x_min, y_min, x_max, y_max] (box bounding cordinates)
    imcrop_bbox_dict_tmp = {k:np.array(v).tolist() for k, v in imcrop_bbox_dict.items()}

    io.save_json(imsize_dict, metadata_dir + 'referit_imsize_dict.json')
    io.save_json(imcrop_dict, metadata_dir + 'referit_imcrop_dict.json')
    io.save_json(query_dict,  metadata_dir + 'referit_query_dict.json')
    io.save_json(imcrop_bbox_dict_tmp, metadata_dir + 'referit_imcrop_bbox_dict.json')

    # imcrop_bbox_dict_tmp = {k:np.array(v).tolist() for k, v in imcrop_bbox_dict.items()}
    # io.save_json(imcrop_bbox_dict_tmp, metadata_dir + 'referit_imcrop_bbox_dict.json')

    # save trainval imcrop list
    trnval_imlist_file = './data/split/referit_trainval_imlist.txt'
    trainval_imcrop_list_save = './data/training/trainval_imcrop_list.txt'
    test_imlist_file = './data/split/referit_test_imlist.txt'
    test_imcrop_list_save = './data/training/test_imcrop_list.txt'

    trainval_im = io.load_str_list(trnval_imlist_file)
    trainval_crops = [imcrop+'.png' for im in trainval_im for imcrop in imcrop_dict[im]]
    with open(trainval_imcrop_list_save, 'w') as f:
        f.writelines('\n'.join(trainval_crops))

    # save test imcrop list
    test_im = io.load_str_list(test_imlist_file)
    test_crops = [imcrop+'.png' for im in test_im for imcrop in imcrop_dict[im]]
    with open(test_imcrop_list_save, 'w') as f:
        f.writelines('\n'.join(test_crops))


if __name__ == '__main__':
    main()