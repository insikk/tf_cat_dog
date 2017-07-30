import numpy as np
import time
import math
import os.path
import read_data
import model_cnn as model
import tensorflow as tf
import eval_cnn

from trainer import Trainer


flags = tf.app.flags


TRAIN_DATA_DIR = 'train_output/'
CHECKPOINT_FILE = 'model.ckpt'
CHECKPOINT_FILE_PATH = os.path.join(TRAIN_DATA_DIR, CHECKPOINT_FILE)

flags.DEFINE_integer("num_epochs", 100, "Num Epochs [100]")
flags.DEFINE_integer("batch_size", 64, "Batch size [60]")
NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES


def run_training(config):
    print("config.num_epochs:", config.num_epochs)
    image, label = read_data.inputs(data_set='train', batch_size=config.batch_size, num_epochs=config.num_epochs)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    train_images, train_labels = tf.train.shuffle_batch(
        [image, label], batch_size=config.batch_size, num_threads=2,
        capacity=1000 + 3 * config.batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    ## Build a model
    m = model.get_model(config, is_training=True)        
    
    ## Build trainer
    trainer = Trainer(config)
    train_op = trainer.training(m.loss)

    ## Summary 

    summary_op = tf.summary.merge_all()

    ## Saver

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

    
    ## Init a model
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    sess = tf.Session()
    sess.run(init_op)

    ## Start input queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    summary_writer = tf.summary.FileWriter(TRAIN_DATA_DIR, sess.graph)

    # TODO: add model loader to continue training, or load pretrained weights. 
    
    # TODO: replace with global_step
    step = 0
    num_iter_per_epoch = int(math.ceil(NUM_TRAIN_EXAMPLES / config.batch_size))
    try:

        while not coord.should_stop():
            start_time = time.time()
            batch_images, batch_labels = sess.run([train_images, train_labels])

            feed_dict = m.get_feed_dict(batch_images, batch_labels)

            _, loss_value = sess.run([train_op, m.loss], feed_dict=feed_dict)

            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                print('Step %d : loss = %.5f (%.3f sec)'
                        % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if step % num_iter_per_epoch == 0 and step > 0: # Do not save for step 0
                num_epochs = int(step / num_iter_per_epoch)
                saver.save(sess, CHECKPOINT_FILE_PATH, global_step=step)
                print('epochs done on training dataset = %d' % num_epochs)
                # eval_cnn.evaluate('validation', checkpoint_dir=TRAIN_DATA_DIR)

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps' % (config.num_epochs, step))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(_):
    config = flags.FLAGS
    run_training(config)


if __name__ == '__main__':
    tf.app.run()