import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import os
import sys
from CDN import Model

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(PROJECT_ROOT,'..'))

import retriever


class RunModel(Model):       
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
                  [query indexes, [bounding box vector (VGG), bounding box spaital features], ..., 
                  [bounding box vector (VGG), bounding box spaital features], index of the true label]
                  
            start/end: batch data is built from data[start:end]
            
        Returns:
            attn_idx: attn_idx[i, j]=1 if the j'th bbox in the i'th query is not padding, else equals to 0. 
            
            padded_queries: list of queries, padded to the length of the longest query in the batch.
                            Note: vocab['p<pad>']=0
                            
            padded_im: list of bounding boxes vectors, padded to the maximum number of bbox per query.
                       Note: padded vector is vector of zeros. 
                            
            padded_bbox: list of bounding boxes spatial features, padded to the maximum number of bbox per query.
                         Note: padded vector is vector of zeros.  
        
            dist_labels: dist_labels[i][j]=1 if j is the true bbox for query i, else dist_labels[i][j]=0
                        
        '''
                      
        qlen = max([len(data[i][0]) for i in range(start, end)]) # Length fo the longest query
        imlen = max([len(data[i]) for i in range(start, end)])-2 # Maximum number of bbox per query.
        padded_queries, padded_im, padded_bbox, attn_idx = [], [], [], []
        
        # Build one hot labels from the labels index, given in the data.                  
        labels = [item[-1] for item in data[start:end]] # data[i][-1]=index of the true bbox of query i
        dist_labels = np.zeros((end-start, imlen)) #label distribution
        dist_labels[[i for i in np.arange(end-start)], [l for l in labels]]=1
        
        im_dim, bbox_dim = data[0][1][0].shape[1], data[0][1][1].shape[1]
        for i in range(start, end):
            padded_queries.append(self.q_padding(data[i][0], qlen))
            
            attn_idx.append([1 for _ in range(len(data[i])-2)]+[0 for _ in range(imlen-(len(data[i])-2))])
            
            padded_im.append(np.concatenate([data[i][j][0] for j in range(1, len(data[i])-1)] + 
                                       [np.full((imlen-(len(data[i])-2), im_dim), self.vocab['<pad>'], dtype=np.float32)], axis=0))
            
            padded_bbox.append(np.concatenate([data[i][j][1] for j in range(1, len(data[i])-1)] + 
                                       [np.full((imlen-(len(data[i])-2),bbox_dim), self.vocab['<pad>'], dtype=np.float32)], axis=0))
           
            
        return np.array(attn_idx), np.array(padded_queries, dtype=np.int32), np.array(padded_im), np.array(padded_bbox), np.array(dist_labels)
            
   
    def ground(self, data=None, start=None, end=None, 
               sess=None, feed_dict = None, scores=[]):
        '''
        Given a query and a list of bboxes, the function returns the index of the chosen bbox and the ground truth bbox.
        
        Params:
            data: A numpy array with datasat's data points
            start/end: The function only take data points from data[start:end]
            imScale: whether to scale the images vectors
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
                attn_idx, padded_queries, padded_im, padded_bbox, labels = self.build_data(
                    data, start, end)
                    
                feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.labels:labels,
                        self.attn_idx:attn_idx}
                    
            if len(scores)==0:
                feed_dict[self.isTrain]=False
                feed_dict[self.dropout_in]=1.
                feed_dict[self.dropout_out]=1.
                feed_dict[self.dropout_img]=1.
                feed_dict[self.dropout_q]=1.
                scores = sess.run(self.scores, feed_dict=feed_dict) # get score for each bbox

        return np.argmax(scores, axis=1), np.argmax(feed_dict[self.labels], axis=1)
        
        
    def iou_accuracy(self, data=None, start=None, end=None, sess=None, 
                     feed_dict=None, threshold=0.5, test=False, scores=[]):
        '''
        Calculate the IOU score between the Model bbox and the true bbox.
        
         Params:
            data: A numpy array with datasat's data points
            start/end: The function only take data points from data[start:end]
            imScale: whether to scale the images vectors
            threshold: If IOU>0.5 this is a true positive
        ''' 
                          
        # Get score for each bbox (labels) and th true bbox index (gt_idx)                  
        labels, gt_idx = self.ground(data, start, end, sess=sess, feed_dict=feed_dict, scores=scores)
        acc = 0
        
        for i in range(start, end):
            gt = data[i][gt_idx[i-start]+1][1][0] # ground truth bbox. Note that len(data)!=len(gt_idx)=batch_size
            crops = np.expand_dims(data[i][labels[i-start]+1][1][0], axis=0) #Model chosen bbox. Note that len(data)!=len(labels)=batch_size
            acc += (retriever.compute_iou(crops, gt)[0]>threshold) #IOU for the i sample.
            
        return acc/(end-start)
        
    def accuracy(self, data=None, start=None, end=None, sess=None, 
                 feed_dict=None, scores=[]):
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
                
            loss = 0
            if len(scores)==0:
                feed_dict[self.isTrain]=False
                feed_dict[self.dropout_in]=1.
                feed_dict[self.dropout_out]=1.
                feed_dict[self.dropout_img]=1.
                feed_dict[self.dropout_q]=1.
                loss, scores = sess.run([self.loss, self.scores], feed_dict=feed_dict)
                    
            acc = sum(
                np.equal(
                    np.argmax(scores, axis=1), 
                    np.argmax(feed_dict[self.labels], axis=1))/len(feed_dict[self.labels])
            )
          
        return loss, scores, acc

    def train(self, trn_data, tst_data, epochs_num,  start_epoch=0, dropout_in=1.,
              dropout_out=1., dropout_img=1., dropout_q=1.):
                          
        '''
        Apart of training the model, this method also gather relevant statistics of the 
        image and language domains.
        
        Params:
            trn_data: list, train set. 
             
            tst_data: list, test set. 
             
            epochs_num: number of epochs
             
            start_epoch: number of first epoch. 
            
            dropout_in: dropout ratio of rnn inputs.
            
            dropout_output: dropout ratio of rnn output.
            
            dropout_img: dropout ratio of images vectors before the last attention layer .
        '''                  
        
        trn_nbatch = len(trn_data)//self.batch_size
        tst_nbatch = len(tst_data)//self.batch_size
        # list to hold accuracy and loss of test and train sets
        self.test_res, self.train_res = [], [] 
        
        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(self.params_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Loading parameters from', ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
            else:
                print('Initializing variables')
                
            qMean_list, qSTD_list, qRange_list, imgMean_list, imgSTD_list, imgRange_list = [], [], [], [], [], []
            Gq, Gimg = [], []
            for epoch in range(start_epoch, epochs_num):
                    
                ############
                # Training #
                ############
                    
                startTime = datetime.now().replace(microsecond=0)   
                print('='*50,'\nTrain, epoch:',epoch)
                np.random.shuffle(trn_data)
                trn_loss, trn_acc, trn_iou = 0, 0, 0
                qMean, qSTD, qRange, imgMean, imgSTD, imgRange = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                g_img, g_rnn = 0.0, 0.0
                
                for b in range(trn_nbatch):
                    attn_idx, padded_queries, padded_im, padded_bbox, labels = self.build_data(trn_data, 
                                                                                        b*self.batch_size, 
                                                                                        (b+1)*self.batch_size)

                    feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.attn_idx:attn_idx,
                        self.labels: labels,
                        self.dropout_in:dropout_in,
                        self.dropout_out:dropout_out,
                        self.dropout_img:dropout_img,
                        self.dropout_q:dropout_q,
                        self.isTrain:True
                    }
                    
                    g_rnn_tmp, g_img_tmp, scores, Urnn, Uatt, loss, lr, gs,  _ = sess.run([self.g_rnn, 
                                                      self.g_img, self.scores, self.Urnn, self.Uatt, 
                                                        self.loss, self.learning_rate, 
                                                        self.global_step, self.optimizer], feed_dict=feed_dict)
                    
                    Urnn_tmp = np.reshape(Urnn, (-1, 200))
                    qMean_tmp = np.mean(np.mean(Urnn_tmp, axis=0))
                    qSTD_tmp = np.mean(np.std(Urnn_tmp, axis=0))
                    qRange_tmp = np.mean(np.max(Urnn_tmp, axis=0) - np.min(Urnn_tmp, axis=0))
                    
                    Uatt_tmp = np.reshape(Uatt[:,:,:4096], (-1, 4096)) 
                    imgSTD_tmp = np.mean(np.std(Uatt_tmp, 0))
                    imgMean_tmp = np.mean(Uatt_tmp)
                    
                    # making sure the padds wont be taken into considiration
                    # idx = the raws number that are not paddings in  Uattn
                    attn_idx_tmp = np.reshape(attn_idx, (-1,))
                    idx = [i for j, i in  enumerate(range(len(attn_idx_tmp))) if attn_idx_tmp[j]!=0]
                    #print(sum(attn_idx_tmp), sum(np.sign(idx)))
                    #idx =  np.arange(len(attn_idx_tmp))*attn_idx_tmp
                    Uatt_tmp = np.reshape(Uatt[:,:,:4096], (-1, 4096))[idx,:]
                    imgSTD_tmp = np.mean(np.std(Uatt_tmp, 0))
                    imgMean_tmp = np.mean(np.mean(Uatt_tmp,0))
                    imgRange_tmp= np.mean(np.max(Uatt_tmp, axis=0) - np.min(imgMean_tmp, axis=0))
                    
                    qMean += qMean_tmp
                    qSTD +=qSTD_tmp
                    qRange += qRange_tmp

                    imgMean += imgMean_tmp
                    imgSTD +=imgSTD_tmp
                    imgRange += imgRange_tmp
                       
                    g_img += g_img_tmp
                    g_rnn += g_rnn_tmp
                    

                    loss, scores, acc = self.accuracy(sess=sess, feed_dict=feed_dict)  
                    iou_acc = self.iou_accuracy(trn_data, b*self.batch_size, (b+1)*self.batch_size,
                                                sess=sess, feed_dict=feed_dict, scores=scores)

                    trn_acc += acc/trn_nbatch
                    trn_loss += loss/trn_nbatch
                    trn_iou += iou_acc/trn_nbatch

                    if b%50==0:
                        print('b:%d'%(b),  
                                ';lr:%.3f'%(lr),
                                ';loss:%.2f'%(loss), ';acc:%.2f'%(acc), 
                                ';iou:%.2f'%(iou_acc),
                                ';qMean:%.2f'%(qMean_tmp),
                                ';qSTD:%.2f'%(qSTD_tmp),
                                ';iMean:%.2f'%(imgMean_tmp),
                                ';iSTD:%.2f'%(imgSTD_tmp),
                                ';qRange:%.2f'%(qRange_tmp),
                                ';iRange:%.2f'%(imgRange_tmp),
                                ';Gq:%.5f'%(g_rnn_tmp*100),
                                ';Gi:%.5f'%(g_img_tmp*100),
                                ';time:', datetime.now().replace(microsecond=0)-startTime)

                print('\n*Tr loss: %.3f'%(trn_loss),';Tr acc: %.3f'%(trn_acc), 
                        ';IOU acc: %.3f'%(trn_iou),  
                        ';qMean:%.3f'%(qMean/trn_nbatch),
                        ';qSTD:%.3f'%(qSTD/trn_nbatch),
                        ';iMean:%.3f'%(imgMean/trn_nbatch),
                        ';iSTD:%.3f'%(imgSTD/trn_nbatch),
                        ';qRange:%.3f'%(qRange/trn_nbatch),
                        ';imgRange:%.3f'%(imgRange/trn_nbatch),
                        ';Gq:%.5f'%(g_rnn*100/trn_nbatch),
                        ';Gi:%.5f'%(g_img*100/trn_nbatch),

                      ';Time:', datetime.now().replace(microsecond=0)-startTime, '\n')

                imgMean_list.append(imgMean/trn_nbatch)
                imgSTD_list.append(imgSTD/trn_nbatch)
                qMean_list.append(qMean/trn_nbatch)
                qSTD_list.append(qSTD/trn_nbatch)
                qRange_list.append(qRange/trn_nbatch)
                imgRange_list.append(imgRange/trn_nbatch)
                Gq.append(g_rnn/trn_nbatch)
                Gimg.append(g_img/trn_nbatch)
               
                self.train_res.append([trn_acc, trn_iou, trn_loss])    
                
                self.saver.save(sess, self.params_dir + "/model.ckpt", global_step=epoch)
                
                ###########
                # Testing #
                ###########
                    
                print('Testing, epoch:',epoch)
                tstTime = datetime.now().replace(microsecond=0)
                tst_loss, tst_acc, tst_iou = 0, 0, 0
                for b in range(tst_nbatch):
                    attn_idx, padded_queries, padded_im, padded_bbox, labels = self.build_data(tst_data,
                                                                                        b*self.batch_size, 
                                                                                        (b+1)*self.batch_size)
                    feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.attn_idx:attn_idx,
                        self.labels: labels,
                        self.dropout_in:1.,
                        self.dropout_out:1.,
                        self.dropout_img:1.,
                        self.dropout_q:1.,
                        self.isTrain:False
                    }
                    scores, loss = sess.run([self.scores, self.loss], feed_dict=feed_dict)
                    _,_, acc = self.accuracy(sess=sess, feed_dict=feed_dict, scores=scores)
                    iou_acc = self.iou_accuracy(
                        tst_data, b*self.batch_size, int(b+1)*self.batch_size, sess=sess, 
                        feed_dict=feed_dict, scores=scores)

                    tst_acc += acc/tst_nbatch
                    tst_loss += loss/tst_nbatch
                    tst_iou += iou_acc/tst_nbatch
                    if b%50==0:
                        print('batch:', b, ';loss: %.3f'%(loss), ';acc: %.3f'%(acc), 
                               ';iou_acc: %.3f'%(iou_acc), ';time:', 
                              datetime.now().replace(microsecond=0)-startTime)
                    
                print('\n*Test loss: %.3f'%(tst_loss), ';Test accuracy %.3f'%(tst_acc), 
                      ';Test IOU: %.3f'%(tst_iou), ';Time:', datetime.now().replace(microsecond=0)-startTime)
                self.test_res.append([tst_acc, tst_iou, tst_loss])
            print('='*50,'\n')
        return self.test_res, self.train_res, imgMean_list, imgSTD_list, imgRange_list, qMean_list, qSTD_list, qRange_list, Gq, Gimg
