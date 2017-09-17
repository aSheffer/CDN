from __future__ import print_function, division
import sys
import os
import numpy as np
import string
import json
from nltk.stem.snowball import SnowballStemmer
import gensim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util import io
import retriever

query_file = './data/metadata/referit_query_dict.json'
w2v_path = './data/metadata/w2v.bin'
vocab_path = './data/metadata/w2v_vocab.json'

query_dict = io.load_json(query_file) # imageName_boxName--> list of imageName_boxName discription
queries_txt = list(query_dict.values()) 

def main():
	# We cut off the punctuations (exclude) and cancatunate 
	# all queries describing the same image (after stemming)
	print("Cleaning text...")
	exclude = set(string.punctuation)
	dataset = []
	for item in queries_txt:
	    txt = []
	    for q in item:
	        clean = ''.join([ch.lower() if ch not in exclude else ' ' for ch in q.strip()]) # clean punctuation
	        txt+=[w for w in clean.split()]
	    dataset.append(txt)

	# W2V
	print('Traingin w2v model...')
	model = gensim.models.Word2Vec(dataset, window=5, min_count=0, size=100)
	model.most_similar('car')

	print("Saving files...")
	# Adding zero vector for padding
	pad = np.array([0.0 for _ in range(model.syn0norm.shape[1])])
	pad = pad.reshape([1, pad.shape[0]])

	vecs = np.concatenate((pad, model.syn0norm))
	np.save(open(w2v_path, 'wb'), vecs)

	vocab = {k:v.index+1 for k,v in model.vocab.items()}
	vocab['<pad>'] = 0


	with open(vocab_path, 'w') as f:
		f.write(json.dumps(vocab))




if __name__ == '__main__':
	main()