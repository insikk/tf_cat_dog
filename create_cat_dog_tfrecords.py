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


flags = tf.app.flags

flags.DEFINE_string("splitname", "trainval", "trainval or test")


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
from PIL import Image
from os import  walk

def read_images(path, classes, img_height = 128, img_width = 128, log=False):
    filenames = next(walk(path))[2]
    num_files = len(filenames)
    
    if log:
        print("Reading images from %s ... %d files are there" % (path, num_files))

    images = []
    labels = []
    for i, filename in enumerate(filenames):
        if i % 2000 == 0:
            if log:
                print("%d / %d" % (i, len(filenames))) 

        img = Image.open(join(path, filename))
        img.thumbnail((img_height, img_width)) # BiCubic Resize
        # img.save(join('thumb', filename)) # save as file
        imgarr = np.array(img).astype(np.uint8)
        if imgarr.shape[2] == 1:
            print("gray! found:", imgarr.shape)
            print("filename::", filename)
        if imgarr.shape[2] == 4:
            print("alpha! found:", imgarr.shape)
            print("filename::", filename)

            imgarr = imgarr[:, :, :3]
            print("remove alpha:", imgarr.shape)
        images.append(imgarr)
        class_name = filename[0:3].lower() # Luckily both 'cat' and 'dog' have 3 characters
        if class_name == 'cat' or class_name == 'dog':
            labels.append(classes.index(class_name))
    if log:
        print("Done!") 

    return images, labels, filenames



def convert_to(images, labels, filenames, name, log=False):
    if log:
        print("converting %s to tfrecords..."%(name))
    num_examples = labels.shape[0]
    if len(images) != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                     (len(images), num_examples))
    

    filename = join(DATA_DIR, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    count = 0
    for index in range(num_examples):
        if index % 2000 == 0:
            if log:
                print("%d / %d" % (index, num_examples)) 
        image = images[index]
        image_raw = image.tobytes()
        rows = image.shape[0]
        cols = image.shape[1]
        depth = image.shape[2]
        shape = np.array(image.shape, np.int32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),   # NOT assuming one-hot format of original data
            'image_id': _bytes_feature(filenames[index].encode('utf-8')),
            'image_raw': _bytes_feature(image_raw),
            'shape': _bytes_feature(shape.tobytes())}))
        writer.write(example.SerializeToString())
        count += 1 
    writer.close()
    if log:
        print("Done!") 
        print("count: ", count)


def process_trainval():
    log_flag = True

    train_images, train_labels, train_filenames = read_images(TRAIN_DATA_PATH, IMG_CLASSES,
                                                        IMG_HEIGHT, IMG_WIDTH, log_flag)
    train_labels = np.array(train_labels)
    
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

    validation_images = [train_images[i] for i in val_idx]
    validation_labels = train_labels[val_idx] # numpy indexing
    validation_filenames = [train_filenames[i] for i in val_idx]
    train_images = [train_images[i] for i in train_idx]
    train_labels = train_labels[train_idx] # numpy indexing
    train_filenames = [train_filenames[i] for i in train_idx]

    # Convert to Examples and write the result to TFRecords.

    convert_to(train_images, train_labels, train_filenames, 'train', log_flag)
    convert_to(validation_images, validation_labels, validation_filenames, 'validation', log_flag)

def process_test():
    log_flag = True
    print("Reading test images...")
    test_images, test_labels, test_filenames = read_images(TEST_DATA_PATH, IMG_CLASSES,
                                                        IMG_HEIGHT, IMG_WIDTH, log_flag)
    test_labels = np.ones(len(test_images)) # dummy label
    convert_to(test_images, test_labels, test_filenames, 'test', log_flag)

def main(_):
    config = flags.FLAGS
    if config.splitname == "trainval":
        process_trainval()
    elif config.splitname == "test":
        process_test()
    else:
        print("unknown splitname is given:", config.splitname)
        print("exit program.")


if __name__ == '__main__':
    tf.app.run()
