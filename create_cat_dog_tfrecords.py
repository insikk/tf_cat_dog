"""
# Create TFRecords from Data

* TF Records are easy way to utilize tensorflow input queue. 
* It reduce burden of file IO. 

Just convert it to TFExample form. 

"""

import numpy as np
from os.path import join
import tensorflow as tf
import read_data



DATA_DIR = 'data/records'
TRAIN_DATA_PATH = 'data/train/'
TEST_DATA_PATH = 'data/test/'


IMG_CLASSES = read_data.IMG_CLASSES
NUM_CLASSES = len(IMG_CLASSES)
IMG_HEIGHT = read_data.IMG_HEIGHT
IMG_WIDTH = read_data.IMG_WIDTH
IMG_CHANNELS = read_data.IMG_CHANNELS
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS
NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES
NUM_VALIDATION_EXAMPLES = read_data.NUM_VALIDATION_EXAMPLES
NUM_TEST_EXAMPLES = read_data.NUM_TEST_EXAMPLES


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


from scipy.misc import imread, imresize
from os import  walk

def read_images(path, classes, img_height = 128, img_width = 128, img_channels = 3, log=False):
    

    filenames = next(walk(path))[2]
    num_files = len(filenames)
    
    if log:
        print("Reading images from %s ... %d files are there" % (path, num_files))

    images = np.zeros((num_files, img_height, img_width, img_channels), dtype=np.uint8)
    labels = np.zeros((num_files, ), dtype=np.uint8)
    for i, filename in enumerate(filenames):
        if i % 2000 == 0:
            if log:
                print("%d / %d" % (i, len(filenames))) 
        img = imread(join(path, filename))
        img = imresize(img, (img_height, img_width)) # Make square image. Discard ratio. 
        images[i, :, :, :] = img
        class_name = filename[0:3].lower() # Luckily both 'cat' and 'dog' have 3 characters
        if class_name == 'cat' or class_name == 'dog':
            labels[i] = classes.index(class_name)
        
    if log:
        print("Done!") 

    return images, labels



def convert_to(images, labels, name, log=False):
    if log:
        print("converting %s to tfrecords..."%(name))
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = join(DATA_DIR, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        if index % 2000 == 0:
            if log:
                print("%d / %d" % (index, num_examples)) 
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),   # NOT assuming one-hot format of original data
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    if log:
        print("Done!") 


def main():
  log_flag = True

  train_images, train_labels = read_images(TRAIN_DATA_PATH, IMG_CLASSES,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, log_flag)

  # Generate a validation set.
  cat_idx = np.where(train_labels == 1)[0]
  print("number of cats:", len(cat_idx))
  dog_idx = np.where(train_labels == 0)[0]
  print("number of dogs:", len(dog_idx))

  cat_val_idx = np.random.choice(cat_idx, 2500, replace=False)
  cat_train_idx = np.setdiff1d(cat_idx, cat_val_idx)

  dog_val_idx = np.random.choice(dog_idx, 2500, replace=False)
  dog_train_idx = np.setdiff1d(dog_idx, dog_val_idx)

  val_idx = np.random.permutation(np.array(cat_val_idx.tolist() + dog_val_idx.tolist()))
  train_idx = np.random.permutation(np.array(cat_train_idx.tolist() + dog_train_idx.tolist()))

  validation_images = train_images[val_idx, :, :, :]
  validation_labels = train_labels[val_idx]
  train_images = train_images[train_idx, :, :, :]
  train_labels = train_labels[train_idx]

  # Convert to Examples and write the result to TFRecords.

  convert_to(train_images, train_labels, 'train', log_flag)
  convert_to(validation_images, validation_labels, 'validation', log_flag)

  print("Reading test images...")
  test_images, test_labels = read_images(TEST_DATA_PATH, IMG_CLASSES,
                                                      IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, log_flag)

  convert_to(test_images, test_labels, 'test', log_flag)


if __name__ == "__main__":
  main()