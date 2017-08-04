
import tensorflow as tf
from tensorflow.contrib import slim

import read_data

NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES

REG_STRENGTH = 0.001
INITIAL_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 0.00001
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 5
MOVING_AVERAGE_DECAY = 0.9999

def _loss_summaries(total_loss):
    losses = tf.get_collection('losses')
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)

class Trainer:
    def __init__(self, config):
        self.config = config

    def training(self, total_loss):

        global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = int(EPOCHS_PER_LR_DECAY * NUM_TRAIN_EXAMPLES / self.config.batch_size)
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LR_DECAY_FACTOR, staircase=True)
        # learning_rate = tf.maximum(learning_rate, MIN_LEARNING_RATE)
        learning_rate = 0.001

        tf.summary.scalar('learning_rate', learning_rate)

        _loss_summaries(total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)        

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        return train_op


    def evaluation(self, logits, true_labels):
        correct_pred = tf.nn.in_top_k(logits, true_labels, 1)
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))*100
