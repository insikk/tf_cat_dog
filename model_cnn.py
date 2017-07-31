from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import read_data
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES
NUM_CLASSES = read_data.NUM_CLASSES
DROP_PROB = 0.5

def get_model(config, is_training=True):
    model = Model(config, is_training)
    return model

def _activation_summary(x):
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


class Model:
    def __init__(self, config, is_training):
        # Build Model
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.is_training = is_training

        # Define forward inputs here
        N = config.batch_size
        self.images = tf.placeholder('float32', [None, 128, 128, 3], name='images')
        self.labels = tf.placeholder('int32', [None], name='labels')

        self.logits = self._inference(self.images)
        self.pred_classes = tf.argmax(tf.nn.softmax(self.logits), axis=1)
        self.acc = tf.metrics.accuracy(self.labels, self.pred_classes)
        

        self.loss_cls = self._loss(self.logits, self.labels)
        tf.add_to_collection('losses', self.loss_cls)

        self.loss_reg = slim.losses.get_regularization_losses()
        tf.add_to_collection('losses', self.loss_reg)

        self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
    def _inference(self, images):

        
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(images, 256, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.flatten(net, scope='flatten5')
            net = slim.fully_connected(net, 256, scope='fc6')
            # Adds a dropout layer to prevent over-fitting.
            net = slim.dropout(net, DROP_PROB, is_training=self.is_training)
            logits = slim.fully_connected(net, NUM_CLASSES, activation_fn=None)
            _activation_summary(logits)

        return logits


    def _loss(self, logits, labels):
        one_hot_labels = slim.one_hot_encoding(
            labels,
            NUM_CLASSES)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels, name='cross_entropy_total')
        data_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        return data_loss

    
    def get_feed_dict(self, images, labels):
        feed_dict = {self.images: images, self.labels: labels}
        return feed_dict
