import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import os
import sys
import retriever
from CDND import Model

trainset_file = './data/training/w2v_train_data_new.bin'
testset_file = './data/training/w2v_test_data_new.bin'
vocab_file =  './data/metadata/w2v_vocab.json'
embed_path =  './data/metadata/w2v.bin'

def load_datasets():
	print('Loading datasets...')
	trainset = np.load(open(trainset_file, 'rb'))
	# delete data points where the query length is zero
	trainset = np.array([item for item in trainset if len(item[0])!=0])

	testset = np.load(open(testset_file, 'rb'))
	# delete data points where the query length is zero
	testset = np.array([item for item in testset if len(item[0])!=0])

	# loading vocabulary 
	with open(vocab_file, 'r') as f:
	    vocab = json.loads(f.read())
	vocab['<unk>'] = len(vocab)

	# Words vectors
	embed_vecs = np.load(open(embed_path, 'rb')).astype(np.float32)

	return vocab, embed_vecs, trainset, testset


def main():
	params_dir = input("Please enter the path to the directory in which you'd like to save the parameters: ")
	ephocs_num = input("For how many ehpocs would you like the model to be trained: ")


	vocab, embed_vecs, trainset, testset = load_datasets()
	tf.reset_default_graph()
	m = Model(
	    params_dir=params_dir,
	    img_dims=trainset[0][1][0].shape[1], 
	    bbox_dims=testset[0][1][1].shape[1], 
	    lr=.05,
	    embed_size=embed_vecs.shape[1])


	m.train(trainset, testset, ephocs_num=int(ephocs_num), dropout_in=0.5, dropout_out=0.5)

if __name__ == "__main__":
	main()