import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import os
import sys

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(PROJECT_ROOT,'..'))

import retriever


embed_path =  '../data/metadata/w2v.bin'
embed_vecs = np.load(open(embed_path, 'rb')).astype(np.float32)



class Model():
    '''
    This is the implementation for CBoWG model. 

    CBoWG's language model uses only a trainable words embeddings. It embeds each query 
    by summing its words vectors and calculates the similarity between the query and each 
    of the candidates bboxes (bounding boxes) via the cosine similarity between their embeddings. 
    The image model remains the same as in all our models. 

    '''
    def __init__(self,
                 batch_size, 
                 
                 #Image's vector size.
                 img_dims, 
                 

                 # where to save Parameters
                 params_dir,
                 #Spaital features length.
                 bbox_dims, 
                 vocab, 

                 lr, #  learning rate.
                 decay_steps, 
                 decay_rate,
                 embed_size=embed_vecs.shape[1],
                 embed_vecs=embed_vecs):
        
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.bbox_dims = bbox_dims 
        self.embed_size = embed_size
        self.vocab = vocab
        self.lr=lr
        self.params_dir=params_dir

        self.queries = tf.placeholder(tf.int32, [None, None], name='queries_holder')
        self.img  = tf.placeholder(tf.float32, [None, None, self.img_dims], name='img_holder')# VGG output vectors
        self.bboxes = tf.placeholder(tf.float32, [None, None, self.bbox_dims], name='bboxes_holder')# spatial bbox's features.

        # attn_idx: inicates whether attention box is a dummy (0) or not (1).
        self.attn_idx = tf.placeholder(tf.float32, [None, None], name='attn_idx')

        self.labels = tf.placeholder(tf.float32, [None, None], name='labels_holder')

        with tf.variable_scope('embed_scope'):
            embed = tf.get_variable(name='embed', initializer=embed_vecs, dtype=tf.float32)
            embed_queries = tf.nn.embedding_lookup(embed, self.queries, name='embed_queries')
                
        # Each query is represent as the normed average of its word vectors
        # shape: batch_size x 1 x embed_size
        avgQ = tf.nn.l2_normalize(tf.reduce_sum(embed_queries, axis=1, keepdims=True), axis=-2)
            
        # Concatinate images vectors and their spaital features
        img_vecs = tf.concat([self.img, self.bboxes], 2)
             
        # Trandorm the images vectors to have embed_size
        # and normelize them.
        img_newVec = tf.nn.l2_normalize(
            tf.reshape(
                self.linear(tf.reshape(img_vecs, [-1, img_dims+bbox_dims]), self.embed_size), 
                shape=[self.batch_size, -1, int(self.embed_size)]),
             axis=-1)
            
        # Calculate cosine distance between
        # each query and all of its bboxes
        dist = tf.reduce_sum(avgQ*img_newVec, axis = -1)
            
        # Calculate the distances masked softmax (we use self.attn_idx to mas
        max_logits = tf.reduce_max(dist, axis=-1)
        masked_logits = tf.exp((dist-tf.expand_dims(max_logits, axis=1))*self.attn_idx)*self.attn_idx
        self.scores = self.attn_idx*masked_logits/(tf.reduce_sum(masked_logits, axis=-1, keepdims=True)+1e-09)
            
            
        # Cross entophy loss.
        self.loss = tf.reduce_mean(
            -tf.reduce_sum(
                self.labels*tf.log(self.scores+0.00000001)+
                     (1-self.labels)*tf.log((1-self.scores)+0.00000001), 
                axis=-1)
        )


        ##############
        # Optimizers #
        ##############

        starter_learning_rate = self.lr
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        decay_steps=decay_steps, 
                                                        decay_rate=decay_rate, staircase=True)

        
        self.optimizer =  tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss, 
                                                               global_step=self.global_step)  

        if not os.path.exists(self.params_dir):
                os.makedirs(self.params_dir)
        self.saver = tf.train.Saver()

        
    def length(self, seq):
        '''
        Retruns real lengths (before addings) of all queries in seq  .
        '''
        return tf.cast(tf.reduce_sum(tf.sign(tf.abs(seq)), reduction_indices=1), tf.int32)
       

    def linear(self, inputs, output_dim, scope='linear', bias=True, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable('W', initializer=tf.contrib.layers.xavier_initializer(),
                                shape=(inputs.get_shape()[-1], output_dim))
            if bias:
                b = tf.get_variable('b', initializer=tf.constant_initializer(0.1),
                               shape=[1, output_dim])
                return tf.matmul(inputs, W) + b
            
            return tf.matmul(inputs, W)

        
    def q_padding(self, seq, max_length):
        '''
        Pad  seq with vocab['<pad>'] (0) to max_length length.
        '''                  
        return seq + [self.vocab['<pad>']]*(max_length-len(seq))

    
    def build_data(self, data, start, end):
        '''
        Build batch.
        ------------
        
        Params:
            data: each entry in this list has the following structure:
                  [query indexes, [bounding box vector, bounding box spaital features], ..., [bounding box vector, bounding box spaital features], index of the true label]
            start/end: batch data is built from data[start:end]
            
        Returns:
            attn_idx: attn_idx[i, j]=1 ifthe j'th bbox in the i'th query is not padding, else equals to 0. 
            
            
            padded_queries: list of queries, padded to the length of the longest query in the batch.
                            Note: vocab['pad']=0
                            
            padded_im: list of bounding boxes vectors, padded to the maximum number of bbox per query.
                            Note: padded vector is vector of zeros. 
                            
            padded_bbox: list of bounding boxes spatial features, padded to the maximum number of bbox per query.
                            Note: padded vector is vector of zeros.  
        
            onehot_labels: onehot_labels[i][j]=1 if j is the true bbox for query i, else  onehot_labels[i][j]=0
            
            addNoise: Boolean. Whether to add normal noise to the images.
                        
        '''
                          
        qlen = max([len(data[i][0]) for i in range(start, end)]) # Length fo the longest query
        imlen = max([len(data[i]) for i in range(start, end)])-2 # Maximum number of bbox per query.
        padded_queries, padded_im, padded_bbox, attn_idx = [], [], [], []
        
        # Build one hot labels from the labels index, given in the data.                  
        labels = [item[-1] for item in data[start:end]] #data[i][-1]=index of the true bbox of query i
        onehot_labels = np.zeros((end-start, imlen))
        onehot_labels[np.arange(end-start), labels]=1
                          
        im_dim, bbox_dim = data[0][1][0].shape[1], data[0][1][1].shape[1]
        for i in range(start, end):
            padded_queries.append(self.q_padding(data[i][0], qlen))
            
            attn_idx.append([1 for _ in range(len(data[i])-2)]+[0 for _ in range(imlen-(len(data[i])-2))])
            
            padded_im.append(np.concatenate([data[i][j][0] for j in range(1, len(data[i])-1)] + 
                                       [np.full((imlen-(len(data[i])-2), im_dim), self.vocab['<pad>'], dtype=np.float32)], axis=0))
            
            padded_bbox.append(np.concatenate([data[i][j][1] for j in range(1, len(data[i])-1)] + 
                                       [np.full((imlen-(len(data[i])-2),bbox_dim), self.vocab['<pad>'], dtype=np.float32)], axis=0))
           
            
        return np.array(attn_idx), np.array(padded_queries, dtype=np.int32), np.array(padded_im), np.array(padded_bbox), np.array(onehot_labels)
            
   
    def ground(self, data=None, start=None, end=None, sess=None, feed_dict = None, isEdit=True):
        '''
        Given a query and a list of bboxes, the function returns the index of the referred bbox.
        '''
        isSess = (sess==None)
        if isSess:
            sess = tf.Session()
        with sess.as_default():
            if isSess:
                tf.global_variables_initializer().run()
                ckpt = tf.train.get_checkpoint_state(self.params_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
                else:
                    print('Initializing variables')
            if feed_dict is None:
                attn_idx, padded_queries, padded_im, padded_bbox, labels = self.build_data(data, start, end)
                feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.labels:labels,
                        self.attn_idx:attn_idx
                    }
            scores = sess.run(self.scores, feed_dict=feed_dict) # get score for each bbox

        return np.argmax(scores, axis=1), np.argmax(feed_dict[self.labels], axis=1)
        
        
    def iou_accuracy(self, data, start, end, sess=None, feed_dict = None, threshold=0.5, test=False, isEdit=True):
        '''
        Calculate the IOU score between the Model bbox and the true bbox.
        ''' 
                          
        # Get score for each bbox (labels) and th true bbox index (gt_idx)                  
        if feed_dict is None:
            labels, gt_idx = self.ground(data, start, end, sess=sess, feed_dict=feed_dict, isEdit=isEdit)
        else: labels, gt_idx = self.ground(sess=sess, feed_dict=feed_dict, isEdit=isEdit)
        acc = 0
        
        for i in range(start, end):
            gt = data[i][gt_idx[i-start]+1][1][0] # ground truth bbox
            crops = np.expand_dims(data[i][labels[i-start]+1][1][0], axis=0) #Model chosen bbox
            acc += (retriever.compute_iou(crops, gt)[0]>threshold) #IOU for the i sample.
            
        return acc/(end-start)
        
    def accuracy(self, data=None, start=None, end=None, sess=None, feed_dict = None, isEdit=True):
        isSess = (sess==None)
        if isSess:
            print('Building sess')
            sess = tf.Session()
        with sess.as_default():
            if isSess:
                print('Building sess used')
                tf.global_variables_initializer().run()
                ckpt = tf.train.get_checkpoint_state(self.params_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print('3')
                    self.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
                else:
                    print('Initializing variables')
            if feed_dict is None:
                print('Building feed_dict')
                attn_idx, padded_queries, padded_im, padded_bbox, labels = self.build_data(data, start, end)
                feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.attn_idx:attn_idx,
                        self.labels:labels,
                    }
            scores = sess.run(self.scores, feed_dict=feed_dict)
            acc = sum(np.equal(np.argmax(scores, axis=1), np.argmax(feed_dict[self.labels], axis=1))/len(feed_dict[self.labels]))

                    
        return acc
    
        
    def train(self, trn_data, tst_data, epochs_num):
                          
        '''
        Params:
             trn_data: list, train set. 
             
             tst_data: list, test set. 
             
             epochs_num: number of epochs
             
             start_epoch: number of first epoch.
             
             edit_reward: int, coefficient to multiply the reward by when editing a word.
             
             startA: int, Start competition only at epoch # startA.
             
             activation_epoch: at epoch numer "activation_epoch", A will be activate.
                               That is, for (activation_epoch-startA) number of epochs, 
                               A will chooce an action randomly.
             
            muteB: After A starts, for each epoch which A & B trains, 
                   only A will be trained for this amount of epochs.
                   
            editProb: robabilty for editing a query.
            
            activateAProb: when running A, we can choos an action randomly or taknig A decision. 
                            This is the starting probabilty for NOT choocing an action ranomdly.
            
            max_activateAProb: Final probabilty for NOT choocing an action ranomdly.
            
            dropout_in: dropout ratio of B's rnn inputs.
            
            dropout_output: dropout ratio of B's rnn output.
            
            dropout_img: dropout ratio of images vectors before the last attention layer .
            
            addNoise: Boolean. Whether to add normal noise to the images (see build_data).
                               
        '''                  
        
        trn_nbatch = len(trn_data)//self.batch_size
        tst_nbatch = len(tst_data)//self.batch_size
        self.test_res, self.train_res = [], [] #list to hold accuracy of test set
        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(self.params_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Loading parameters from', ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
            else:
                print('Initializing variables')
                
            for epoch in range(epochs_num):
                startTime = datetime.now().replace(microsecond=0)
                    
                print('='*50,'\nTrain, epoch:',epoch)
                np.random.shuffle(trn_data)
                trn_loss, trn_acc, trn_iou = 0, 0, 0
              
                for b in range(trn_nbatch):
                    attn_idx, padded_queries, padded_im, padded_bbox, labels = self.build_data(trn_data, 
                                                                                        b*self.batch_size, (b+1)*self.batch_size)

                    feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.attn_idx:attn_idx,
                        self.labels: labels
                    }

                  
                    loss, lr, gs, _ = sess.run([self.loss, self.learning_rate, self.global_step, self.optimizer], feed_dict=feed_dict)

                    acc = self.accuracy(sess=sess, feed_dict=feed_dict)  
                    iou_acc = self.iou_accuracy(
                        trn_data, b*self.batch_size, (b+1)*self.batch_size, 
                        sess=sess, feed_dict=feed_dict)
                    
                    
                    trn_loss += loss/trn_nbatch
                    trn_acc += acc/trn_nbatch
                    trn_iou += iou_acc/trn_nbatch

                    if b%100==0:
                        print('epoch:',epoch, ';batch:', b, 
                              ';gs:', gs, ';lr: %.4f'%(lr), ';loss: %.2f'%(loss), 
                              ';acc: %.3f'%(acc), ';iou: %.3f'%(iou_acc), ';time:', datetime.now().replace(microsecond=0)-startTime)    
                  
                print('\n*Train loss: %.3f'%(trn_loss),                                                                                            
                          ';Train accuracy: %.3f'%(trn_acc),  ';IOU accuracy: %.3f'%(trn_iou), 
                          ';Time:', datetime.now().replace(microsecond=0)-startTime, '\n')
                self.train_res.append([trn_acc, trn_iou, trn_loss])
                    
                self.saver.save(sess, self.params_dir + "/model.ckpt", global_step=epoch)    
                
                
                print('Testing, epoch:',epoch)
                tstTime = datetime.now().replace(microsecond=0)
                tst_loss, tst_acc, tst_iou = 0, 0, 0
                for b in range(tst_nbatch):
                    attn_idx, padded_queries, padded_im, padded_bbox, labels = self.build_data(
                                                                                tst_data,
                                                                                b*self.batch_size, 
                                                                                (b+1)*self.batch_size)
                    
                    feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.attn_idx:attn_idx,
                        self.labels: labels
                    }
                    
                    loss = sess.run(self.loss, feed_dict=feed_dict)

                    acc = self.accuracy(sess=sess, feed_dict=feed_dict)
                    iou_acc = self.iou_accuracy(
                        tst_data, b*self.batch_size, (b+1)*self.batch_size, sess=sess, feed_dict=feed_dict)
                    
                    tst_acc += acc/tst_nbatch
                    tst_loss += loss/tst_nbatch
                    tst_iou += iou_acc/tst_nbatch
                    if b%100==0:
                        print('Batch:', b, ';loss: %.3f'%(loss), ';acc: %.3f'%(acc), 
                               ';iou_acc: %.3f'%(iou_acc), ';time:', datetime.now().replace(microsecond=0)-startTime)
                        
                print('\n*Test loss: %.3f'%(tst_loss), ';Test accuracy %.3f'%(tst_acc), 
                      ';IOU accuracy: %.3f'%(tst_iou), ';Time:', datetime.now().replace(microsecond=0)-startTime)
                self.test_res.append([tst_acc, tst_iou, tst_loss])
                
            print('='*50,'\n')
        return self.test_res, self.train_res
