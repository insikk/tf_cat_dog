from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from os.path import join
import tensorflow as tf

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'

DATA_DIR = 'data/records/'                     # Local CPU


IMG_CLASSES = ['dog', 'cat']
NUM_CLASSES = 2

IMG_HEIGHT = 280
IMG_WIDTH = 280
IMG_CHANNELS = 3

IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS


NUM_FILES_DATASET = 25000
VALIDATION_SET_FRACTION = 0.2
NUM_TRAIN_EXAMPLES = int((1 - VALIDATION_SET_FRACTION) * NUM_FILES_DATASET)
NUM_VALIDATION_EXAMPLES = int((VALIDATION_SET_FRACTION) * NUM_FILES_DATASET)
NUM_TEST_EXAMPLES = 5000


# This function is not being used
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'shape': tf.FixedLenFeature([], tf.string),
            'image_id': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    shape = tf.decode_raw(features['shape'], tf.int32)
    image_id = tf.cast(features['image_id'], tf.string)

    # image_id = tf.cast(features['image_id'], tf.string)
    img_height = tf.cast(features['height'], tf.int32)
    img_width = tf.cast(features['width'], tf.int32)
    img_depth = tf.cast(features['depth'], tf.int32)
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, shape)


    return image, label, image_id


def inputs(data_set, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
    train: Selects between the train , validation and test data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        file = TRAIN_FILE
    elif data_set == 'validation':
        file = VALIDATION_FILE
    elif data_set == 'test':
        file = TEST_FILE
    else:
        raise ValueError('data_set should be one of \'train\', \'validation\' or \'test\'')
    filename = join(DATA_DIR, file)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label, image_id = read_and_decode(filename_queue)

    return image, label, image_id
