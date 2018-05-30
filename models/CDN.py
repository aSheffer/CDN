import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import os
import sys
from ALSTM import ALSTM


embed_path =  '../data/metadata/w2v.bin'
embed_vecs = np.load(open(embed_path, 'rb')).astype(np.float32)


class Model():
    '''
    This is the implementation for SG+CDN (Supervised GroundeR with Cross Domain Normalization), 
    however, it also offers a method for adding a BN (Batch Normalization) over the image and/or 
    language models (see bnorm_attention function). In addition, you can use attention method 
    in order to test the model without CND nor BN.

    This implementation is meant to test how CDN and BN affect the image and language statistics 
    and update rates in order to analyse the effect of CDS (Cross Domain Statistics). Therefore,
    it also calculate the the domains update rates. 
    '''
    def __init__(self,
                 batch_size, 
                 num_hidden, 
                 
                 #Image's vector size.
                 img_dims, 
                 
                 #Spaital features length.
                 bbox_dims, 
                 vocab, 
                 lr, # learning rate.
                 decay_steps, 
                 decay_rate, 

                 # where to save Parameters
                 params_dir, 
                 
                 # whether to use batch normaliztion/CDN layers over the image/language/both models
                 bnorm=False,
                 CDN=False,
                 embed_size=embed_vecs.shape[1],
                 
                 # Whether to uses prefix levlel attention or not.
                 use_wordAttn=False,
                 
                 # Whther to use bidirectional rnn
                 useBidirectionalRnn=False,
                 
                 # Urnn_norm: Whether to use batch normalization for the queries.
                 # Uatt_norm: Whether to use batch normalization for the VGG outputs.
                 Urnn_norm=True, 
                 Uatt_norm=True,

                # We scale the VGG16 LN outputs by IMGscale
                 IMGscale=1,
                 # We scale the language LN outputs by Qscale
                 Qscale=1,
                 ):
        
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.bbox_dims = bbox_dims 
        self.num_hidden = num_hidden
        self.embed_size = embed_size
        self.vocab = vocab
        self.lr = lr
        self.params_dir=params_dir
        self.IMGscale=IMGscale
        self.Qscale=Qscale
        
        self.queries = tf.placeholder(tf.int32, [None, None], name='queries')
        self.img  = tf.placeholder(tf.float32, [None, None, self.img_dims], name='img')# VGG output vectors
        self.bboxes = tf.placeholder(tf.float32, [None, None, self.bbox_dims], name='bboxes')# spatial bbox's features.

        # attn_idx: inicates whether attention box is a pad (0) or not (1).
        self.attn_idx = tf.placeholder(tf.float32, [None, None], name='attn_idx')
        self.labels = tf.placeholder(tf.float32, [None, None], name='labels')
        
        # Dropout ratio for rnn's inputs and outpouts
        self.dropout_in = tf.placeholder(tf.float32, name='dropoutIn_holder')
        self.dropout_out = tf.placeholder(tf.float32, name='dropoutOut_holder')

        # Dropout ratio for attention vector (for the final attention layer before the loss function)
        self.dropout_img = tf.placeholder(tf.float32, name='dropoutImg_holder')
        # Dropout ratio for query vector (for the final attention layer before the loss function)
        self.dropout_q = tf.placeholder(tf.float32, name='dropoutImg_holder')

        self.isTrain = tf.placeholder(tf.bool, name='isTrain_holder') 
        self.queries_lens = self.length(self.queries) # list of all the lengths of the batch's queriey 

        # Concatenate images vectors and their spatial features. 
        # These vectors will be used for attention when
        # we calculate the loss function.
        attn_vecs = tf.concat([self.img, self.bboxes], 2) 
        voc_size = len(self.vocab)

        # Load pre-trained word embeddings.
        # w2v_embed is not trainable.
        with tf.variable_scope('w2v'):
            w2v_embed = tf.get_variable('w2v_embed', initializer=embed_vecs, trainable=False)
            w2v_queries = tf.nn.embedding_lookup(w2v_embed, self.queries, name='w2v_queries')

        with tf.variable_scope('embed'):
            embed = tf.get_variable('embed', shape=[voc_size, self.embed_size], 
                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            embed_queries_tmp = tf.nn.embedding_lookup(embed, self.queries, name='embed_queries')

        embed_queries = embed_queries_tmp+w2v_queries

        with tf.variable_scope('rnn'):
            if use_wordAttn:
                normed_image = self.IMGscale*tf.contrib.layers.layer_norm(self.img, begin_norm_axis=2)
                attn_vecs = tf.concat([normed_image, self.bboxes], 2) 
                
            cell = ALSTM(num_units=self.num_hidden, 
                    img_attn_dim=self.img_dims+self.bbox_dims,
                    img_attn_states=attn_vecs,
                    img_attn_idx=self.attn_idx,
                    batch_size=self.batch_size, 
                    dropout_in=self.dropout_in, dropout_out=self.dropout_out)

            if useBidirectionalRnn:
                self.outputs, self.last_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    dtype=tf.float32,
                    sequence_length=self.queries_lens,
                    inputs=embed_queries)

                state = tf.concat([self.last_states[0][1], self.last_states[1][1]], -1)
            
            else:
                self.outputs, self.last_states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float32,
                    sequence_length=self.queries_lens,
                    inputs=embed_queries)
                state = self.last_states[1]
        
        if CDN:
            self.scores = self.CDN_attention(state) 
        elif bnorm: # If using batch normalization 
            self.scores = self.bnorm_attention(state, Urnn_norm=Urnn_norm, Uatt_norm=Uatt_norm) 
        else:
            self.scores = self.attention(state) 

        # Cross entophy loss for each of the queries in the batch.
        self.loss = tf.reduce_mean(-tf.reduce_sum(
                        self.labels*tf.log(self.scores+0.00000001)+
                            (1-self.labels)*tf.log((1-self.scores)+0.00000001), 
                        axis=-1))
            
        # Language (g_rnn) and image (g_img) domains update rates.
        self.g_rnn =  tf.reduce_mean(tf.norm(tf.gradients(self.loss, self.Urnn)[0], axis=1))
        self.g_img =  tf.reduce_mean(
            tf.reduce_sum(
                tf.norm(
                    tf.gradients(self.loss, self.Uatt)[0]+0.0000000001, axis=2), axis=1)/tf.reduce_sum(self.attn_idx, 1))
        

        ##############
        # Optimizers #
        ##############

        starter_learning_rate = self.lr
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, 
                                                        self.global_step,
                                                        decay_steps=decay_steps, 
                                                        decay_rate=decay_rate, 
                                                        staircase=True)

        self.optimizer =  tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)  

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
            W1 = tf.get_variable('W1', initializer=tf.contrib.layers.xavier_initializer(),
                                shape=(inputs.get_shape()[-1], output_dim))
            if bias:
                b = tf.get_variable('b', initializer=tf.constant_initializer(0.1),
                               shape=[1, output_dim])
                return tf.matmul(inputs, W1) + b
            
            return tf.matmul(inputs, W1)

            
    def pad_img(self, attn_vecs):
        '''
        Each query atteched to a different number of BBOXes, hence, 
        we've padded each query BBOXes candidates with zeros (in attn_vecs). When using 
        batch normlization, this might effect the mean and std which the BN calculates. Therefore, 
        we change the paddings s.t it won't affect the statistics. This is done 
        by:
        
            1. Let the i-th features set contain the i-th feature of each BBOX
               in the batch. We first calculate the std and mean of all the 
               values in this set which are not paddings.
               
            2. We build a random tensor (bbox_padded) with the shape of attn_vecs s.t in 
               its i features set, the sub-set of the values that correspond 
               to the padded values (in attn_vecs) will have the same mean and 
               std as calculated in 1. 
               
            3. We than replace only the padded features in attn_vecs by their correspond
               features in bbox_padded.
            
        It's easy to show mathematicly that batch normelizing the resulted tensor will have 
        the same effect as batch normelizing attn_vecs.
            
            
        Params:
            attn_vecs: bboxes vectors.
            
        Returns:
            new_attn_vecs: see above. 
            
        '''
        
        # Calculating the statistics of each features set
        bbox_dim = self.img_dims+self.bbox_dims
        attnVecs = tf.reshape(attn_vecs, [-1, bbox_dim])
        
        # mask[i,j]=1 if the j-th BBOX in the i-th query is paddings, 
        # else its zero.
        mask = tf.reshape(self.attn_idx, [-1,1])
        mean = tf.reduce_sum(attnVecs, 0, keepdims=True)/tf.reduce_sum(mask, axis=0, keepdims=True)
        std = tf.sqrt(
            tf.reduce_sum(
                mask*(attnVecs-mean)**2/tf.reduce_sum(mask, axis=0, keepdims=True),
                axis=0,
                keepdims=True)
        )
        
        
        # Building bbox_padded (see comment above)
        rand = tf.random_normal(shape=tf.shape(attnVecs))
        mean_r = tf.reduce_sum((1-mask)*rand, 0, keepdims=True)/tf.reduce_sum(1-mask+0.00000001, 0, keepdims=True)
        std_r = tf.sqrt(
            tf.reduce_sum(
                (rand-mean_r)**2/tf.reduce_sum(1-mask+0.00000001, axis=0, keepdims=True),
                axis=0,
                keepdims=True)
        )
        
        
        pad = std*(rand-mean_r)/(std_r+0.00000001)+mean
        bbox_padded = (1-mask)*pad+mask*attnVecs
        new_attn_vecs = tf.reshape(bbox_padded, tf.shape(attn_vecs))
        
        return new_attn_vecs
    
    
    def attention(self, q_embed):
        '''
        Given RNN's output vector, calculate the bboxes scores.
        That is, calculate:
        
                probs = softmax(relu(context(Sq+Satt+b)))
        
        Where:
        Sq = <Wq, queries_states>
        Sattn = <Wattn, attention_bboxes_vectors>
        
        The  bounding box with the highest attention score will be chosen as the correct bounding box.
        
        Params:
            q_embed: Tensor of shape (batch size x num_hidden). RNN's outputs. 
            
        Returns:
            probs: Tensor of shape (batch_size x max bbox number for query).
                   Attention score for each bbox.
        '''
        # concatenate img vectors with spaical features
        attn_vecs = tf.concat([self.img, self.bboxes], 2)
        self.Urnn = q_embed
        
        # Attention vectors, 
        # shape: (batch size x max bbox number for query x attention vector size)
        self.Uatt = attn_vecs
           
        with tf.variable_scope('l1'):
            b = tf.get_variable(
                    'b', 
                    initializer=tf.constant_initializer(0.1), 
                    shape=[1, self.num_hidden])

            context = tf.get_variable(
                    'context', 
                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1), 
                    shape=[self.num_hidden, 1])


            Sq = tf.nn.dropout(
                self.linear(self.Urnn, self.num_hidden, bias=False, scope='Sq'), 
                self.dropout_q)
            
            Sattn = tf.nn.dropout(
                tf.reshape(
                    self.linear(
                        tf.reshape(self.Uatt, (-1, self.img_dims+self.bbox_dims)), 
                        self.num_hidden, 
                        bias=False, scope='Sattn'), 
                    [self.batch_size, -1, self.num_hidden]),
                self.dropout_img)

        out = tf.nn.relu(tf.expand_dims(Sq, 1) + Sattn + b)
        logits = tf.reshape(tf.matmul(tf.reshape(out, (-1, tf.shape(out)[-1])),  context), (tf.shape(out)[0], -1))

        # Calculate logits's masked softmax. We use self.attn_idx for 
        # masking the padded BBOXes.
        max_logits = tf.reduce_max(logits*self.attn_idx, axis=-1, keepdims=True)
        masked_logits = tf.exp((logits-max_logits)*self.attn_idx)*self.attn_idx 
        probs = masked_logits/(tf.reduce_sum(masked_logits, axis=-1, keepdims=True)+1e-09)
        return probs

    
    def bnorm_attention(self, q_embed, Urnn_norm=True, Uatt_norm=True):
        '''
        Given RNN's output vector, calculate the bboxes scores using
        Batch Normalization. That is, calculate:
        
        probs = softmax(relu(context(Sq+Satt+b)))
        
        Where:
        Sq = <Wq, queries_states>
        Sattn = <Wattn, attention_bboxes_vectors>
        
        The  bounding box with the highest attention score will be chosen as the correct bounding box.
        This function uses batch normalization. 
        
        Params:
            q_embed: Tensor of shape (batch size x num_hidden) queries embeddings. 
            Urnn_norm: Whether to use batch normalization for the queries.
            Uatt_norm: Whether to use batch normalization for the VGG outputs.
            
        Returns:
            probs: Tensor of shape (batch_size x max bbox number for query).
                   Attention score for each bbox.
        '''

        if Urnn_norm:
            # RNN's outputs with bath normalization. 
            # Shape: (batch size x num_hidden)
            self.Urnn = tf.contrib.layers.batch_norm(
                q_embed, center=True, scale=True, epsilon=0.000001, is_training=self.isTrain) 
        else:
            self.Urnn = q_embed
        
        if Uatt_norm:
            # Concatenate img vectors with with spaical features
            attn_vecs = self.pad_img(tf.concat([self.img, self.bboxes], 2))
            
            # Attention vectors with bath normalization. 
            # Shape: (batch size x max bbox number for query x attention vector size)
            self.Uatt = tf.contrib.layers.batch_norm(
                attn_vecs, center=True, scale=True, is_training=self.isTrain)
        else:
            # Concatenate img vectors with with spaical features
            attn_vecs = tf.concat([self.img, self.bboxes], 2)
            self.Uatt = attn_vecs
           
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope('bnorm_l1') as scope:
                b = tf.get_variable(
                        'b', 
                        initializer=tf.constant_initializer(0.1), 
                        shape=[1, self.num_hidden])

                context = tf.get_variable(
                        'context', 
                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1), 
                        shape=[self.num_hidden, 1])


                Sq = tf.nn.dropout(
                    self.linear(self.Urnn, self.num_hidden, bias=False, scope='Sq'), 
                    self.dropout_q)
                
                Sattn = tf.nn.dropout(
                            tf.reshape(
                                self.linear(
                                    tf.reshape(self.Uatt, (-1, self.img_dims+self.bbox_dims)), 
                                    self.num_hidden, 
                                    bias=False, scope='Sattn'), 
                                 [self.batch_size, -1, self.num_hidden]),
                            self.dropout_img)
                
                self.AttnVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
                    
                   
            out = tf.nn.relu(tf.expand_dims(Sq, 1) + Sattn + b)
            logits = tf.reshape(tf.matmul(tf.reshape(out, (-1, tf.shape(out)[-1])),  context), (tf.shape(out)[0], -1))

            # Calculate logits's masked softmax. We use self.attn_idx for 
            # masking the padded BBOXes.
            max_logits = tf.reduce_max(logits*self.attn_idx, axis=-1, keepdims=True)
            masked_logits = tf.exp((logits-max_logits)*self.attn_idx)*self.attn_idx 
            probs = masked_logits/(tf.reduce_sum(masked_logits, axis=-1, keepdims=True)+1e-09)
            return probs


    def CDN_attention(self, q_embed):
        '''
        Given the queries embeddings, calculate the bboxes scores using CDN.
        That is, calculate:
        
        probs = softmax(relu(<context, Sq+Satt+b>))
        
        Where:
        Sq = <Wq, queries_states>
        Sattn = <Wattn, attention_bboxes_vectors>
        
        The  bounding box with the highest attention score will be chosen as the correct bounding box.
        
        Params:
            q_embed: Tensor of shape (batch size x num_hidden), queries embeddings. 
            
        Returns:
            probs: Tensor of shape (batch_size x max bbox number for query).
                   Attention score for each bbox.
        '''

        self.Urnn = self.Qscale*tf.contrib.layers.layer_norm(q_embed) 
        UattTmp = self.IMGscale*tf.contrib.layers.layer_norm(self.img, begin_norm_axis=2)
        self.Uatt = tf.concat([UattTmp, self.bboxes], 2)
        with tf.variable_scope('CDNAttn') as scope:
            b = tf.get_variable(
                    'b', 
                    initializer=tf.constant_initializer(0.1), 
                    shape=[1, self.num_hidden])

            context = tf.get_variable(
                    'context', 
                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1), 
                    shape=[self.num_hidden, 1])


            Sq = tf.nn.dropout(
                self.linear(self.Urnn, self.num_hidden, bias=False, scope='Sq'), 
                self.dropout_q)

            Sattn = tf.nn.dropout(
                        tf.reshape(
                            self.linear(
                                tf.reshape(self.Uatt, (-1, self.img_dims+self.bbox_dims)), 
                                self.num_hidden, 
                                bias=False, scope='Sattn'), 
                             [self.batch_size, -1, self.num_hidden]),
                        self.dropout_img)

            out = tf.nn.relu(tf.expand_dims(Sq, 1) + Sattn + b)
            logits = tf.reshape(tf.matmul(tf.reshape(out, (-1, tf.shape(out)[-1])),  context), (tf.shape(out)[0], -1))

            # Calculate logits's masked softmax. We use self.attn_idx for 
            # masking the padded BBOXes.
            max_logits = tf.reduce_max(logits*self.attn_idx, axis=-1, keepdims=True)
            masked_logits = tf.exp((logits-max_logits)*self.attn_idx)*self.attn_idx 
            probs = masked_logits/(tf.reduce_sum(masked_logits, axis=-1, keepdims=True)+1e-09)

            return probs