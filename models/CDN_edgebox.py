import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import sys
import json

base_dir = os.path.dirname(os.path.realpath(__file__))+'/../'
sys.path.append(base_dir)
sys.path.append(base_dir+'models/')

import retriever
from ALSTM import ALSTM

vocab_file =  base_dir+'data/metadata/w2v_vocab.json'
embed_path =  base_dir+'data/metadata/w2v.bin'


class Model():
    def __init__(self,
                 params_dir,
                 batch_size, 
                 num_hidden=200, 

                 #Image's vector size.
                 img_dims=4096, 
                 
                 #Spaital features length.
                 bbox_dims=8, 
                 lr=0.05, # learning rate.
                 decay_steps=10000, 
                 decay_rate=0.9, 
                 embed_size=100,

                 # whether to use batch normaliztion/CDN layers over the image/language/both models
                 bnorm=False,
                 CDN=False,
                 
                 # Whether to uses prefix levlel attention or not.
                 use_wordAttn=False,

                 # Urnn_norm: Whether to use batch normalization for the queries.
                 # Uatt_norm: Whether to use batch normalization for the VGG outputs.
                 Urnn_norm=True, 
                 Uatt_norm=True,
                 
                 # Whther to use bidirectional rnn
                 useBidirectionalRnn=False,
                 
                 # We scale the VGG16 LN outputs by IMGscale
                 IMGscale=1,
                 # We scale the language LN outputs by Qscale
                 Qscale=1
                 ):

        self.batch_size = batch_size
        self.img_dims = img_dims
        self.bbox_dims = bbox_dims 
        self.num_hidden = num_hidden
        self.embed_size = embed_size
        self.lr = lr
        self.IMGscale=IMGscale
        self.Qscale=Qscale
        self.params_dir = params_dir
        self.vocab, embed_vecs = self.load_datasets()
        
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

        # Concatinate images vectors and their spaital features. 
        # These vectors wlll be used for attenionn when 
        # we calculate the loss function.
        attn_vecs = tf.concat([self.img, self.bboxes], 2) 
        voc_size = len(self.vocab)

        # Load pre-trained word imaddings.
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


        #Cross entophy loss for each of the queries in the batch.
        self.loss = tf.reduce_mean(-tf.reduce_sum(
                        self.labels*tf.log(self.scores+0.00000001), 
                        axis=-1))

        
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

    def load_datasets(self):
        # loading vocabulary 
        with open(vocab_file, 'r') as f:
            vocab = json.loads(f.read())
        vocab['<unk>'] = len(vocab)

        # Words vectors
        embed_vecs = np.load(open(embed_path, 'rb')).astype(np.float32)

        return vocab, embed_vecs
    
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
        Urnn = q_embed
        
        # Attention vectors, 
        # shape: (batch size x max bbox number for query x attention vector size)
        Uatt = attn_vecs
           
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
                self.linear(Urnn, self.num_hidden, bias=False, scope='Sq'), 
                self.dropout_q)
            
            Sattn = tf.nn.dropout(
                tf.reshape(
                    self.linear(
                        tf.reshape(Uatt, (-1, self.img_dims+self.bbox_dims)), 
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
            Urnn = tf.contrib.layers.batch_norm(
                q_embed, center=True, scale=True, epsilon=0.000001, is_training=self.isTrain) 
        else:
            Urnn = q_embed
        
        if Uatt_norm:
            # Concatenate img vectors with with spaical features
            attn_vecs = self.pad_img(tf.concat([self.img, self.bboxes], 2))
            
            # Attention vectors with bath normalization. 
            # Shape: (batch size x max bbox number for query x attention vector size)
            Uatt = tf.contrib.layers.batch_norm(
                attn_vecs, center=True, scale=True, is_training=self.isTrain)
        else:
            # Concatenate img vectors with with spaical features
            attn_vecs = tf.concat([self.img, self.bboxes], 2)
            Uatt = attn_vecs
           
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
                    self.linear(Urnn, self.num_hidden, bias=False, scope='Sq'), 
                    self.dropout_q)
                
                Sattn = tf.nn.dropout(
                            tf.reshape(
                                self.linear(
                                    tf.reshape(Uatt, (-1, self.img_dims+self.bbox_dims)), 
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
        Given the queries embeddings, calculate the attention over all the query's 
        bounding boxes vectors using CDN, That is, calculate:
        
        probs = softmax(relu(context(Sq+Satt+b)))
        
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

        # Shape: (batch size x num_hidden)
        Urnn = self.Qscale*tf.contrib.layers.layer_norm(q_embed) 
        UattTmp = self.IMGscale*tf.contrib.layers.layer_norm(self.img, begin_norm_axis=2)
        Uatt = tf.concat([UattTmp, self.bboxes], 2)
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
                self.linear(Urnn, self.num_hidden, bias=False, scope='Sq'), 
                self.dropout_q)

            Sattn = tf.nn.dropout(
                        tf.reshape(
                            self.linear(
                                tf.reshape(Uatt, (-1, self.img_dims+self.bbox_dims)), 
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
  
        
    def q_padding(self, seq, max_length):
        '''
        Pad  seq with vocab['<pad>'] (0) to max_length length.
        '''     
        newSeqs = []   
        for s in seq:
            newSeqs.append(s + [self.vocab['<pad>']]*(max_length-len(s)))
        return newSeqs

    
    def build_data(self, data, start, end):
        '''
        Build batch.
        ------------
        
        Params:
            data: each entry in this list has the following structure:
                  [query indexes, [bounding box vector (VGG), bounding box spaital features], ..., 
                  [bounding box vector (VGG), bounding box spaital features], labels]
                  
            start/end: batch data is built from data[start:end]
            
        Returns:
            attn_idx: attn_idx[i, j]=1 if the j'th bbox in the i'th query is not padding, else equals to 0. 
            
            padded_queries: list of queries, padded to the length of the longest query in the batch.
                            Note: vocab['p<pad>']=0
                            
            padded_im: list of bounding boxes vectors, padded to the maximum number of bbox per query.
                       Note: padded vector is vector of zeros. 
                            
            padded_bbox: list of bounding boxes spatial features, padded to the maximum number of bbox per query.
                         Note: padded vector is vector of zeros.  
        
            dist_labels: dist_labels[i][j]=iou/sum_iou, if iou between j and gt > 0.5 (sum_iou = sum of iou of all bbox with iou >=0.5)
                         else, dist_labels[i][j]=0.
                        
        '''
                      
        qlen = max([len(data[i][0]) for i in range(start, end)]) # Length fo the longest query
        imlen = 100 # Maximum number of bbox per query.
        padded_queries, padded_im, padded_bbox, attn_idx = [], [], [], []
        
        # Build one hot labels from the labels index, given in the data.                  
        labels = [item[-1] for item in data[start:end]] # data[i][-1]=index of the true bbox of query i
        dist_labels = np.zeros((end-start, imlen)) #label distribution
        for j, item in enumerate(labels):
            for l in item:
                dist_labels[j][l[0]] = l[1]
        
        im_dim, bbox_dim = data[0][1][0].shape[1], data[0][1][1].shape[1]
        for i in range(start, end):
            padded_queries.append(self.q_padding(data[i][0], qlen))
            
            
            padded_im.append(np.concatenate([data[i][j][0] for j in range(1, len(data[i])-1)] + 
                                       [np.full((imlen-(len(data[i])-2), im_dim), self.vocab['<pad>'], dtype=np.float32)], axis=0))
            
            padded_bbox.append(np.concatenate([data[i][j][1] for j in range(1, len(data[i])-1)] + 
                                       [np.full((imlen-(len(data[i])-2),bbox_dim), self.vocab['<pad>'], dtype=np.float32)], axis=0))
           
            
        return np.array(attn_idx), np.array(padded_queries, dtype=np.int32), np.array(padded_im), np.array(padded_bbox), np.array(dist_labels)
            
   
    def iou_accuracy(self, sess=None, feed_dict = None, scores=[]):
        '''
        Given a query and a list of bboxes, the function returns the index of the chosen bbox and the ground truth bbox.
        
        Params:
            data: A numpy array with datasat's data points
            start/end: The function only take data points from data[start:end]
            imScale: whether to scale the images vectors
        '''

        loss = 0 
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
                    
            if len(scores)==0:
                feed_dict[self.isTrain]=False
                feed_dict[self.dropout_in]=1.
                feed_dict[self.dropout_out]=1.
                feed_dict[self.dropout_img]=1.
                feed_dict[self.dropout_q]=1.
                loss, scores = sess.run([self.loss, self.scores], feed_dict=feed_dict) # get score for each bbox

        # indexes of bboxes which has an iou>=0.5 with the ground truth 
        gt = np.argwhere(feed_dict[self.labels]>0)
        # indexes of the highest scored bbox per image
        pred = np.argmax(scores, axis=-1)

        # gt_per_image_list[i] = list bboxes' indexes whose IOU with the ground truth are higher than 0.5
        gt_per_image_list = [[] for item in range(self.batch_size)]
        for item in gt:
            gt_per_image_list[item[0]].append(item[1])
            
        iou_acc = 0
        for i, item in enumerate(pred):
            if item in gt_per_image_list[i]:
                iou_acc+=1
                
        return loss, iou_acc/self.batch_size
        
        
    def predict(self, queries, img, bboxes, attn_idx):    
        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(self.params_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Loading parameters from', ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
            else:
                print('Initializing variables')
            
            feed_dict = {
                    self.queries:queries,
                    self.img:img,
                    self.bboxes:bboxes,
                    self.dropout_in:1.,
                    self.dropout_out:1.,
                    self.dropout_img:1.,
                    self.dropout_q:1.,
                    self.isTrain:False,
                    self.attn_idx:attn_idx,
                    self.labels:[[]]
            }
            
            scores = sess.run(self.scores, feed_dict=feed_dict)
        return scores

    def train(self, trn_path, tst_path, epochs_num,  start_epoch=0, dropout_in=1.,
              dropout_out=1., dropout_img=1., dropout_q=1.):
                          
        '''
        Params:
             trn_path: path to directory which holds the train set batches files. 

             tst_path: path to directory which holds the test set batches files. 
             
             tst_data: list, test set. 
             
             epochs_num: number of epochs
             
             start_epoch: number of first epoch. 
            
            dropout_in: dropout ratio of rnn inputs.
            
            dropout_output: dropout ratio of rnn output.
            
            dropout_img: dropout ratio of images vectors before the last attention layer.
            
            toTest: Whether or not to test the algo during training. 
                                                  
        '''                
        trn_nbatch = len(os.listdir(trn_path)) # number of train batches (in trn_path directory)
        tst_nbatch = len(os.listdir(tst_path)) # number of test batches (in tst_path directory)
        print('Number of train batches:', trn_nbatch)
        print('Number of test batches:', tst_nbatch)

        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(self.params_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Loading parameters from', ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
            else:
                print('Initializing variables')
                
            train_batches = list(range(trn_nbatch))
            trn_res, tst_res = [], []
            for epoch in range(start_epoch, epochs_num):
                    
                ############
                # Training #
                ############
                    
                startTime = datetime.now().replace(microsecond=0)   
                print('='*50,'\nTrain, epoch:',epoch)
                trn_loss, trn_iou = 0, 0
                np.random.shuffle(train_batches)
                bidx=0
                for b in train_batches:
                    trn = np.load(open(trn_path+str(b)+'.bin', 'rb'))
                    attn_idx, padded_queries, padded_im, padded_bbox, labels = trn[:,0], trn[:,1], trn[:,2], trn[:,3], trn[:,4]

                    attn_idx = np.concatenate([np.expand_dims(item,0) for item in trn[:,0]], 0)
                    padded_im = np.concatenate([np.expand_dims(item,0) for item in trn[:,2]], 0)
                    padded_bbox = np.concatenate([np.expand_dims(item,0) for item in trn[:,3]], 0)

                    # Build one hot labels from the labels index, given in the data.                  
                    dist_labels = np.zeros((self.batch_size, 100)) #label distribution
                    for j, item in enumerate(trn[:,4]):
                        for l in item:
                            dist_labels[j][l[0]] = l[1]

                    # padding queries
                    qlen = max([len(q) for q in trn[:,1]])
                    padded_queries = self.q_padding(padded_queries, qlen)
            

                    feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.attn_idx:attn_idx,
                        self.labels: dist_labels,
                        self.dropout_in:dropout_in,
                        self.dropout_out:dropout_out,
                        self.dropout_img:dropout_img,
                        self.dropout_q:dropout_q,
                        self.isTrain:True
                    }
                    
                    lr, gs,  _ = sess.run([self.learning_rate, 
                                           self.global_step, self.optimizer], feed_dict=feed_dict)
         
                    loss, iou_acc = self.iou_accuracy(sess=sess, feed_dict=feed_dict)

                    trn_loss += loss/trn_nbatch
                    trn_iou += iou_acc/trn_nbatch

                    if bidx%50==0:
                        print('b:%d'%(bidx),  
                                ';lr:%.3f'%(lr),
                                ';loss:%.2f'%(loss), 
                                ';iou:%.2f'%(iou_acc),
                                ';time:', datetime.now().replace(microsecond=0)-startTime)
                    bidx+=1

                print('\n*Tr loss: %.3f'%(trn_loss), 
                        ';IOU acc: %.3f'%(trn_iou),  ';Time:', datetime.now().replace(microsecond=0)-startTime, '\n')
                trn_res.append([trn_iou, trn_loss])       
                self.saver.save(sess, self.params_dir + "/model.ckpt", global_step=epoch)
            
                ###########
                # Testing #
                ###########

                print('Testing, epoch:',epoch)
                tstTime = datetime.now().replace(microsecond=0)
                tst_loss, tst_iou = 0, 0
                for b in range(tst_nbatch):
                    tst = np.load(open(tst_path+str(b)+'.bin', 'rb'))
                    attn_idx, padded_queries, padded_im, padded_bbox, labels = tst[:,0], tst[:,1], tst[:,2], tst[:,3], tst[:,4]

                    attn_idx = np.concatenate([np.expand_dims(item,0) for item in tst[:,0]], 0)
                    padded_im = np.concatenate([np.expand_dims(item,0) for item in tst[:,2]], 0)
                    padded_bbox = np.concatenate([np.expand_dims(item,0) for item in tst[:,3]], 0)

                    # Build one hot labels from the labels index, given in the data.                  
                    dist_labels = np.zeros((self.batch_size, 100)) #label distribution
                    for j, item in enumerate(tst[:,4]):
                        for l in item:
                            dist_labels[j][l[0]] = l[1]

                    # padding queries
                    qlen = max([len(q) for q in tst[:,1]])
                    padded_queries = self.q_padding(padded_queries, qlen)
            

                    feed_dict = {
                        self.queries:padded_queries,
                        self.img:padded_im,
                        self.bboxes:padded_bbox,
                        self.attn_idx:attn_idx,
                        self.labels: dist_labels,
                        self.dropout_in:1.,
                        self.dropout_out:1.,
                        self.dropout_img:1.,
                        self.dropout_q:1.,
                        self.isTrain:True
                    }
                    
                    loss, scores = sess.run([self.loss, self.scores], feed_dict=feed_dict)
         
                    _, iou_acc = self.iou_accuracy(sess=sess, feed_dict=feed_dict, scores=scores)

                    tst_loss += loss/tst_nbatch
                    tst_iou += iou_acc/tst_nbatch

                    if b%50==0:
                        print('b:%d'%(b),  
                              ';loss:%.2f'%(loss), 
                              ';iou:%.2f'%(iou_acc),
                              ';time:', datetime.now().replace(microsecond=0)-startTime)

                    
                print('\n*Test loss: %.3f'%(tst_loss), ';Test IOU: %.3f'%(tst_iou), 
                    ';Time:', datetime.now().replace(microsecond=0)-startTime)
                tst_res.append([tst_iou, tst_loss])     

            return tst_res, trn_res
