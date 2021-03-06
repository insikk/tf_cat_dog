from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import read_data
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES
NUM_CLASSES = read_data.NUM_CLASSES

def get_model(config, is_training=True):
    model = Model(config, is_training)
    return model

def _activation_summary(x):
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=2,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
  """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = slim.conv2d(inputs, 64, [3, 3], 2, padding='VALID',
                                scope='conv1', activation_fn=None)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='conv1_bn')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')

            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_2', activation_fn=None)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='conv2_bn')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_3', activation_fn=None)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='conv3_bn')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool3')

            net = slim.conv2d(net, 256, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_2')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_3', activation_fn=None)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='conv4_bn')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool4')
            
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_3', activation_fn=None)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='conv5_bn')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [3, 3], padding='VALID',
                                scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                biases_initializer=tf.zeros_initializer(),
                                scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points

class Model:
    def __init__(self, config, is_training):
        # Build Model
        self.config = config
        self.is_training = is_training

        # Define forward inputs here
        N = config.batch_size
        self.images = tf.placeholder('float32', [None, 256, 256, 3], name='images')
        self.labels = tf.placeholder('int32', [None], name='labels')

        self.logits = self._inference(self.images)
        self.pred_classes = tf.cast(tf.argmax(tf.nn.softmax(self.logits), axis=1), tf.int32)
        self.pred_dog = tf.nn.softmax(self.logits)[:, 0]
        self.acc= slim.metrics.accuracy(self.labels, self.pred_classes)
        

        self.loss_cls = self._loss(self.logits, self.labels)
        tf.add_to_collection('losses', self.loss_cls)

        self.loss_reg = tf.add_n(slim.losses.get_regularization_losses())
        tf.add_to_collection('losses', self.loss_reg)

        self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
    def _inference(self, images):
        with slim.arg_scope(alexnet_v2_arg_scope()):
            logits, end_points = alexnet_v2(images, self.config.num_classes, self.is_training, self.config.keep_prob)
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
