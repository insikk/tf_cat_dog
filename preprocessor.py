import tensorflow as tf



def _smallest_size_at_least(height, width, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale + 0.5) # make sure it is exactly the smallest size. 
  new_width = tf.to_int32(width * scale + 0.5)
  return new_height, new_width


def preprocess(image, augmentation=False, is_training=False):
  """ 
  Input image resize and augmentation 

  image: A image tensor shape of [?, ?, 3] height, weight may vary
  """

  ## input image resize. make smallest size at least 128. keep aspect ratio  
  #image = tf.Print(image, [tf.shape(image)], "preprocess image shape: ")
  ih, iw = tf.shape(image)[0], tf.shape(image)[1]    
  new_ih, new_iw = _smallest_size_at_least(ih, iw, 128)  
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
  image = tf.squeeze(image, axis=[0])
  # image = tf.Print(image, [tf.shape(image)], "preprocess resized image shape: ")

  POST_IMAGE_HEIGHT = 128
  POST_IMAGE_WIDTH = 128
  if is_training:
    if augmentation:
      # 10% chance, use input as-is. 90%, do random crop
      coin_crop = tf.to_float(tf.random_uniform([1]))[0]
      image = tf.cond(tf.greater_equal(coin_crop, 0.1), 
                    lambda: tf.random_crop(image, [POST_IMAGE_HEIGHT, POST_IMAGE_WIDTH, 3]),
                    lambda: tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [POST_IMAGE_HEIGHT, POST_IMAGE_WIDTH]), 0))
    
      # Because these operations are not commutative, randomizing
      # the order their operation.
      coin_order = tf.to_float(tf.random_uniform([1]))[0]
      image = tf.cond(tf.greater_equal(coin_order, 0.5), 
                    lambda: tf.image.random_contrast(tf.image.random_brightness(image, max_delta=63), lower=0.2, upper=1.8),
                    lambda: tf.image.random_brightness(tf.image.random_contrast(image, lower=0.2, upper=1.8), max_delta=63))

      image = tf.image.random_flip_left_right(image)
    else:
      # Just resize. No augmentation
      image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [POST_IMAGE_HEIGHT, POST_IMAGE_WIDTH]), 0)

  else:
    image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [POST_IMAGE_HEIGHT, POST_IMAGE_WIDTH]), 0)
  
  ## Image normalization
  ## TODO: consider subtract mean image and divde by std val over whole train dataset. 
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(image)
  return tf.reshape(float_image, [POST_IMAGE_HEIGHT, POST_IMAGE_WIDTH, 3])
