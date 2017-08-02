import tensorflow as tf

def preprocess(image, augmentation=False, is_training=False):
  """ 
  Input image resize and augmentation 

  image: A image tensor shape of [280, 280, 3]
  """
  if is_training:
    if augmentation:
      image = tf.image.random_flip_left_right(image)
      image = tf.random_crop(image, [256, 256, 3])
    else:
      # Just resize. No augmentation
      image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [256, 256]), 0)


  else:
    # resize 280x280 image to 256x256
    image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [256, 256]), 0)
  
  ## Image normalization
  ## TODO: consider subtract mean image and divde by std val over whole train dataset. 
  # Convert from [0, 255] -> [-0.5, 0.5] floats. 
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image
