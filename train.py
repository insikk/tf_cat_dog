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




flags.DEFINE_string("train_dir", "train_output", "training output directory. checkpoints, logs are saved in here.")

flags.DEFINE_integer("load_step", 0, "load model weight at the given global step [0]")
flags.DEFINE_integer("num_epochs", 100, "Num Epochs [100]")
flags.DEFINE_integer("batch_size", 100, "Batch size [100]")
flags.DEFINE_integer("num_classes", 2, "Num classes [2]")

flags.DEFINE_float("keep_prob", 0.8, "Dropout keep probability [0.8]")

NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES
NUM_VALIDATION_EXAMPLES = read_data.NUM_VALIDATION_EXAMPLES

# TODO: add debug flag so we can debug code without waiting for epoch.
def restore(config, sess):
     """restore from the previous checkpoint"""
    try:
        checkpoint_path = tf.train.latest_checkpoint(config.train_dir)
        restorer = tf.train.Saver()
        restorer.restore(sess, checkpoint_path)
        print ('restored previous model %s from %s'\
                %(checkpoint_path, config.train_dir))
        time.sleep(2)
        return
    except:
        print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
                % (config.train_dir, checkpoint_path))
        print('Start from new weights')
        time.sleep(2)

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


    image_val, label_val = read_data.inputs(data_set='validation', batch_size=config.batch_size, num_epochs=config.num_epochs)

    # We run this in two threads to avoid being a bottleneck.
    val_images, val_labels = tf.train.batch(
        [image, label], batch_size=config.batch_size, num_threads=2,
        capacity=500 + 3 * config.batch_size)

    ## Build a model
    m = model.get_model(config, is_training=True)        
    
    ## Build trainer
    trainer = Trainer(config)
    train_op = trainer.training(m.total_loss)

    ## Summary 

    summary_op = tf.summary.merge_all()

    tf_global_step = slim.get_or_create_global_step()

    # Load from previous one
    if config.load_step > 0:
        save_path = os.path.join(config.train_dir, "{}-{}".format('model.ckpt', config.load_step))
        print("Loading saved model from {}".format(save_path))
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
    else:
        restore(sess)
    
    ## Init a model
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    sess = tf.Session()
    sess.run(init_op)

    ## Saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)

    ## Start input queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(TRAIN_DATA_DIR, sess.graph)

    # TODO: add model loader to continue training, or load pretrained weights. 
    
    # TODO: replace with global_step
    step = 0
    num_iter_per_epoch = int(math.ceil(NUM_TRAIN_EXAMPLES / config.batch_size))
    num_batch_eval = int(math.ceil(NUM_VALIDATION_EXAMPLES / config.batch_size))
    # if True or config.debug:
    #    num_iter_per_epoch = 50
    try:

        while not coord.should_stop():
            start_time = time.time()
            batch_images, batch_labels = sess.run([train_images, train_labels])

            feed_dict = m.get_feed_dict(batch_images, batch_labels)

            global_step, _, loss_cls, loss_reg = sess.run([tf_global_step, train_op, m.loss_cls, m.loss_reg], feed_dict=feed_dict)
            global_step += 1
            tot_loss = loss_cls + loss_reg
            duration = time.time() - start_time
            assert not np.isnan(tot_loss), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                print('GlobalStep %d : loss_cls = %.5f, loss_reg = %.5f, loss_tot = %.5f (%.3f sec)'
                        % (global_step, loss_cls, loss_reg, tot_loss, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if step % num_iter_per_epoch == 0 and step > 0: # Do not save for step 0
                num_epochs = int(step / num_iter_per_epoch)
                CHECKPOINT_FILE_PATH = os.path.join(config.train_dir, 'model.ckpt')
                saver.save(sess, CHECKPOINT_FILE_PATH, global_step=global_step)
                print("Start Periodical Evaluation with Validation Set on Training Graph")
                sum_loss = 0.0
                sum_acc = 0.0

                for batch_idx in range(num_batch_eval): # TODO: change it to tqdm. show progress bar
                    start_time = time.time()
                    batch_images, batch_labels = sess.run([val_images, val_labels])
                    feed_dict = m.get_feed_dict(batch_images, batch_labels)
                    loss_cls, acc, pred_y = sess.run([m.loss_cls, m.acc, m.pred_classes], feed_dict=feed_dict)
                    sum_loss += loss_cls
                    sum_acc += acc
                    duration = time.time() - start_time
                    # print('Eval Batch %d/%d. loss_cls = %.5f, acc = %.2f'%(batch_idx, num_batch_eval, loss_cls, acc))
                print('Eval Done. loss_cls = %.5f, acc = %.2f'%(sum_loss / num_batch_eval, sum_acc / num_batch_eval))
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
