import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import os
import sys
from ALSTM import ALSTM



class Model():
    '''
    This model tests SG woth no textual information.
    '''
    def __init__(self,
                 batch_size, 
                 num_hidden, 
                 
                 #Image's vector size.
                 img_dims, 
                 
                 #Spaital features length.
                 bbox_dims, 
                 lr, #  B's learning rate.
                 decay_steps, 
                 decay_rate, 
                
                 # whether or not to standardize the image features
                 useCDN,
                 vocab,

                 # where to save Parameters
                 params_dir,

                 # We scale the VGG16 LN outputs by IMGscale
                 IMGscale=1,
                 
                 # Whether or not we use spatial features
                 use_spatial=True
                 ):
        
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.vocab = vocab
        self.img_dims = img_dims
        self.bbox_dims = bbox_dims 
        self.lr = lr
        self.use_spatial = use_spatial
        self.IMGscale = IMGscale
        self.params_dir=params_dir

        self.img  = tf.placeholder(tf.float32, [None, None, self.img_dims], name='img')# VGG output vectors
        self.bboxes = tf.placeholder(tf.float32, [None, None, self.bbox_dims], name='bboxes')# spatial bbox's features.
        self.labels = tf.placeholder(tf.float32, [None, None], name='labels')
        
        # attn_idx: inicates whether attention box is a dummy (0) or not (1).
        self.attn_idx = tf.placeholder(tf.float32, [None, None], name='attn_idx')
        
        # Concatinate images vectors and their spaital features. 
        # These vectors wlll be used for attention when 
        # we calculate the loss function.
        attn_vecs = tf.concat([self.img, self.bboxes], 2) 

        if useCDN: # If using batch normalization 
            self.scores = self.CDN_attention() 
        else:
            self.scores = self.attention() 


        # Cross entophy loss for each of the queries in the batch.
        self.loss = tf.reduce_mean(-tf.reduce_sum(
                        self.labels*tf.log(self.scores+0.00000001)+
                            (1-self.labels)*tf.log((1-self.scores)+0.00000001), 
                        axis=-1))

        ##############
        # Optimizers #
        ##############

        starter_learning_rate = self.lr
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

        self.optimizer =  tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)  

        if not os.path.exists(self.params_dir):
                os.makedirs(self.params_dir)
        self.saver = tf.train.Saver()


    def linear(self, inputs, output_dim, scope='linear', bias=True, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            W1 = tf.get_variable('W1', initializer=tf.contrib.layers.xavier_initializer(),
                                shape=(inputs.get_shape()[-1], output_dim))
            if bias:
                b = tf.get_variable('b', initializer=tf.constant_initializer(0.1),
                               shape=[1, output_dim])
                return tf.matmul(inputs, W1) + b
            
            return tf.matmul(inputs, W1)

            
    
    def attention(self):
        '''
        Given the bboxes candidates for each image, calculate the
        probability for each bbox by: 
             
             probs = softmax(relu(<context, Satt+b>))
        
        Where Sattn = <Wattn, attention_bboxes_vectors> 
                     
        Returns:
            probs: Tensor of shape (batch_size x max bbox number for query).
                   Score for each bbox.
        '''
        # concatenate img vectors with spaical features
        # Attention vectors, shape: (batch size x max bbox number for query x attention vector size)
        if self.use_spatial:
            W = tf.concat([self.img, self.bboxes], 2)
            img_dim = self.img_dims+self.bbox_dims
        else:
            W = self.img
            img_dim = self.img_dims
            
        with tf.variable_scope('l1'):
            b = tf.get_variable(
                    'b', 
                    initializer=tf.constant_initializer(0.1), 
                    shape=[1, self.num_hidden])

            context = tf.get_variable(
                    'context', 
                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1), 
                    shape=[self.num_hidden, 1])


            Sattn = tf.reshape(
                    self.linear(
                        tf.reshape(W, (-1, img_dim)), 
                        self.num_hidden, 
                        bias=False, scope='Sattn'), 
                    [self.batch_size, -1, self.num_hidden])
            
        out = tf.nn.relu(Sattn + b)
        logits = tf.reshape(
            tf.matmul(tf.reshape(out, (-1, tf.shape(out)[-1])),  context), 
            (tf.shape(out)[0], -1)
        )

        # Calculate logits's masked softmax. We use self.attn_idx for 
        # masking the padded BBOXes.
        max_logits = tf.reduce_max(logits*self.attn_idx, axis=-1, keepdims=True)
        masked_logits = tf.exp((logits-max_logits)*self.attn_idx)*self.attn_idx 
        probs = masked_logits/(tf.reduce_sum(masked_logits, axis=-1, keepdims=True)+1e-09)
        return probs

    
    def CDN_attention(self):
        '''
        
        Given the bboxes candidates for each image, calculate the
        probability for each bbox by: 
        
        probs = softmax(relu(<context, Sq+Satt+b>))
        
        Where:
        Sq = <Wq, queries_states>
        Sattn = <Wattn, attention_bboxes_vectors>
        
        This function adds LN (layer normalization) layer over the VGG16 outputs. The 
        LN outputs are scaled by self.IMGscale. The bbox with the highest attention score 
        will be chosen as the ground truth bounding box.
            
        Returns:
            probs: Tensor of shape (batch_size x max bbox number for query).
                   Attention score for each bbox.
        '''
            
        UattTmp = self.IMGscale*tf.contrib.layers.layer_norm(self.img, begin_norm_axis=2)
        
        # concatenate img vectors with spaical features if self.use_spatial is true
        if self.use_spatial:
            Uatt = tf.concat([UattTmp, self.bboxes], 2)
            img_dim = self.img_dims+self.bbox_dims
        else:
            Uatt = UattTmp
            img_dim = self.img_dims
           
        with tf.variable_scope('CDNAttn'):
            b = tf.get_variable(
                    'b', 
                    initializer=tf.constant_initializer(0.1), 
                    shape=[1, self.num_hidden])

            context = tf.get_variable(
                    'context', 
                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1), 
                    shape=[self.num_hidden, 1])


            Sattn = tf.reshape(
                    self.linear(
                        tf.reshape(Uatt, (-1, img_dim)), 
                        self.num_hidden, 
                        bias=False, scope='Sattn'), 
                    [self.batch_size, -1, self.num_hidden])
        
            out = tf.nn.relu(Sattn + b)
            logits = tf.reshape(
                tf.matmul(tf.reshape(out, (-1, tf.shape(out)[-1])),  context), 
                (tf.shape(out)[0], -1)
            )


            # Calculate logits's masked softmax. We use self.attn_idx for 
            # masking the padded BBOXes.
            max_logits = tf.reduce_max(logits*self.attn_idx, axis=-1, keepdims=True)
            masked_logits = tf.exp((logits-max_logits)*self.attn_idx)*self.attn_idx 
            probs = masked_logits/(tf.reduce_sum(masked_logits, axis=-1, keepdims=True)+1e-09)

            return probs