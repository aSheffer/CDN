import tensorflow as tf
import numpy as np
import sys 
import os 
import argparse
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('./models/')
from CDN_edgebox import Model

trnset_path =  base_dir + '/data/train_edgbox_batches/'
tstset_path = base_dir + '/data/test_edgbox_batches/'
params_dir_tmp = base_dir + '/data/training/models/edgeBox/'

bsize = 50 #batch size

def runModel(useCDND=False, useBiCDN=False):
	if useCDND: 
		print('Training CDND...')
		tf.reset_default_graph()
		params_dir=params_dir_tmp+'cdnd'
		m = Model(batch_size=bsize, params_dir=params_dir, lr=0.05, CDN=True, IMGscale=0.023, Qscale=0.16)
		tst, trn = m.train(trn_path=trnset_path, tst_path=tstset_path,  start_epoch=0, epochs_num=100, dropout_in=0.5, dropout_out=0.5)

	elif useBiCDN:
		print('Training BiCDN...')
		tf.reset_default_graph()
		params_dir=params_dir_tmp+'=bicdn'
		m = Model(batch_size=bsize, params_dir=params_dir, lr=0.05, CDN=True, IMGscale=0.023, Qscale=0.16, useBidirectionalRnn=True)
		tst, trn = m.train(trn_path=trnset_path, tst_path=tstset_path,  start_epoch=0, epochs_num=100, dropout_in=0.5)

	else:
		print('Training CDN...')
		tf.reset_default_graph()
		params_dir=params_dir_tmp+'cdn'
		m = Model(batch_size=bsize, params_dir=params_dir, lr=0.05, CDN=True, IMGscale=0.023, Qscale=0.16)
		tst, trn = m.train(trn_path=trnset_path, tst_path=tstset_path,  start_epoch=0, epochs_num=100)





def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-CDND", action="store", dest='useCDND', default=False)
	parser.add_argument("-BiCDN", action="store", dest='useBiCDN',  default=False)
	results = parser.parse_args()

	runModel(results.useCDND, results.useBiCDN)

if __name__=='__main__':
	main()