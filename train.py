import numpy as np
import time
import math
import os.path
import read_data
import model_cnn_simple_128 as model

import tensorflow as tf
from tensorflow.contrib import slim

import eval_cnn
import pandas as pd

from trainer import Trainer
import preprocessor



flags = tf.app.flags




flags.DEFINE_string("train_dir", "train_output", "training output directory. checkpoints, logs are saved in here.")

flags.DEFINE_string("mode", "train", "run mode. train, eval")
flags.DEFINE_string("eval_dir", "eval_output", "evaluation output directory. checkpoints, logs are saved in here.")


flags.DEFINE_integer("load_step", 0, "load model weight at the given global step [0]")
flags.DEFINE_integer("num_epochs", 100, "Num Epochs [100]")
flags.DEFINE_integer("batch_size", 200, "Batch size [100]")
flags.DEFINE_integer("num_classes", 2, "Num classes [2]")

flags.DEFINE_float("keep_prob", 0.8, "Dropout keep probability [0.8]")

flags.DEFINE_integer("augmentation", True, "Input image augmentation (flip, random crop) [True]")


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
    image = preprocessor.preprocess(image, config.augmentation, is_training=True)
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    train_images, train_labels = tf.train.shuffle_batch(
        [image, label], batch_size=config.batch_size, num_threads=4,
        capacity=1000 + 3 * config.batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)


    image_val, label_val = read_data.inputs(data_set='validation', batch_size=config.batch_size, num_epochs=config.num_epochs)
    image_val = preprocessor.preprocess(image_val, False, is_training=False)

    # We run this in two threads to avoid being a bottleneck.
    val_images, val_labels = tf.train.batch(
        [image, label], batch_size=config.batch_size, num_threads=4,
        capacity=500 + 3 * config.batch_size)

    ## Build a model
    m = model.get_model(config, is_training=True)        
    
    ## Build trainer
    trainer = Trainer(config)
    train_op = trainer.training(m.total_loss)

    ## Summary 
    summary_op = tf.summary.merge_all()

    tf_global_step = slim.get_or_create_global_step()

    
    ## Init a model
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init_op)

    # Load from previous one
    if config.load_step > 0:
        save_path = os.path.join(config.train_dir, "{}-{}".format('model.ckpt', config.load_step))
        print("Loading saved model from {}".format(save_path))
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
    else:
        restore(config, sess)
    
    ## Saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)

    ## Start input queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(os.path.join(config.train_dir, "summary"), sess.graph)

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

            if global_step % 10 == 0:
                print('GlobalStep %d : loss_cls = %.5f, loss_reg = %.5f, loss_tot = %.5f (%.3f sec/step)'
                        % (global_step, loss_cls, loss_reg, tot_loss, duration))                

            if global_step % num_iter_per_epoch == 0 and global_step > 0: # Do not save for step 0                
                num_epochs = int(global_step / num_iter_per_epoch)
                
                print('epochs done on training dataset = %d. Save checkpoint and write summary' % num_epochs)
                CHECKPOINT_FILE_PATH = os.path.join(config.train_dir, 'model.ckpt')
                saver.save(sess, CHECKPOINT_FILE_PATH, global_step=global_step)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step)

                print("Start Periodical Evaluation with Validation Set on Training Graph")
                sum_loss = 0.0
                sum_acc = 0.0

                for batch_idx in range(num_batch_eval): # TODO: change it to tqdm. show progress bar
                    start_time = time.time()
                    batch_images, batch_labels = sess.run([val_images, val_labels])
                    feed_dict = m.get_feed_dict(batch_images, batch_labels)
                    loss_cls, acc, pred_y, pred_dog = sess.run([m.loss_cls, m.acc, m.pred_classes, m.pred_dog], feed_dict=feed_dict)
                    print("gtlabel[:10]", batch_labels[:10])
                    print("pred_y[:10]", pred_y[:10])
                    print("pred_dog[:10]", pred_dog[:10])
                    sum_loss += loss_cls
                    sum_acc += acc
                    duration = time.time() - start_time
                    # print('Eval Batch %d/%d. loss_cls = %.5f, acc = %.2f'%(batch_idx, num_batch_eval, loss_cls, acc))
                val_loss = sum_loss / num_batch_eval
                val_acc = sum_acc / num_batch_eval
                summary = tf.Summary()
                summary.value.add(tag='val_loss', simple_value=val_loss)
                summary.value.add(tag='val_acc', simple_value=val_acc)
                summary_writer.add_summary(summary, global_step)
                print('Eval Done. loss_cls = %.5f, acc = %.2f'%(sum_loss / num_batch_eval, sum_acc / num_batch_eval))
                
                # eval_cnn.evaluate('validation', checkpoint_dir=TRAIN_DATA_DIR)

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps' % (config.num_epochs, global_step))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def run_eval(config):
    print("EVALUATION MODE !!!!!!!")
    image, label, image_id = read_data.inputs(data_set='test', batch_size=config.batch_size, num_epochs=1)
    image = preprocessor.preprocess(image, config.augmentation, is_training=False)
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    test_images, test_labels, test_ids = tf.train.batch(
        [image, label, image_id], batch_size=config.batch_size, num_threads=4,
        capacity=1000 + 3 * config.batch_size, allow_smaller_final_batch=True)

    ## Build a model
    m = model.get_model(config, is_training=False)
    
    ## Summary 
    summary_op = tf.summary.merge_all()

    tf_global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)

    
    ## Init a model
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init_op)

    # Load from previous one
    if config.load_step > 0:
        # At eval, load trained weight from train_dir
        save_path = os.path.join(config.train_dir, "{}-{}".format('model.ckpt', config.load_step))
        print("Loading saved model from {}".format(save_path))
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
    else:
        restore(config, sess)
    
    ## Saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)

    ## Start input queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_iter_per_epoch = int(math.ceil(NUM_TRAIN_EXAMPLES / config.batch_size))
    num_batch_eval = int(math.ceil(NUM_VALIDATION_EXAMPLES / config.batch_size))
    
    eval_step = 0
    # TODO: see what happend when there is last batch, less then batch size. Handle it. 
    y_test = []
    id_test = []
    try:
        sum_loss = 0.0
        sum_acc = 0.0
        num_examples = 0

        while not coord.should_stop():
            start_time = time.time()
            batch_images, batch_labels, batch_ids = sess.run([test_images, test_labels, test_ids])
            print("batch_ids type:", type(batch_ids))
            print("batch_image shape:", batch_images.shape)
           
            # if batch_images.get_shape()[0] is None:
            #     # handle for the last batch
            #     pad_num = tf.shape(batch_images)[0]
            batch_size = batch_images.shape[0]
            num_examples += batch_size

            feed_dict = m.get_feed_dict(batch_images, batch_labels)
            
            loss_cls, loss_reg, acc, pred_y, pred_dog = sess.run([m.loss_cls, m.loss_reg, m.acc, m.pred_classes, m.pred_dog], feed_dict=feed_dict)
            print("pred_y[:10]", pred_y[:10])
            tot_loss = loss_cls + loss_reg
            sum_loss += loss_cls * batch_size
            sum_acc += acc * batch_size
            y_test += pred_dog.tolist()
            id_test += batch_ids.tolist()
                    
            eval_step += 1
            duration = time.time() - start_time
            print("eval_step: %d, processed: %d, time:%.2f/sec"%(eval_step, eval_step * config.batch_size, duration))
            assert not np.isnan(tot_loss), 'Model diverged with loss = NaN'
    except tf.errors.OutOfRangeError:
        print('Out of range')
    finally:
        test_loss = sum_loss / num_examples
        test_acc = sum_acc / num_examples
        print('Eval Done. loss_cls = %.5f, acc = %.2f'%(test_loss, test_acc))
        # Create samplesubmission file
        print("we got %d predictions:", len(y_test))
        is_kaggle=False
        if is_kaggle:
            subm = pd.read_csv("sample_submission.csv")
            for i, y in enumerate(y_test):
                if y < 0.05:
                    y = 0.05
                if y > 0.95:
                    y = 0.95
                # clip to prevent huge penalty of logloss for wrong label
                subm.loc[i, "label"] = y
            subm.to_csv(os.path.join(config.eval_dir, "submission.csv"), index=False)
        else:
            subm = pd.DataFrame.from_dict({'filename': id_test, 'label': y_test})
            for i, y in enumerate(y_test):
                if y > 0.5:
                    y = 1 # hard prediction for real contest
                else:
                    y = 0
                subm.loc[i, "label"] = int(y)
                subm.loc[i, "filename"] = subm.loc[i, "filename"].decode()
            df = subm
            df[['label']] = df[['label']].astype(int)

            df['sort'] = df['filename'].str.extract('(\d+)', expand=False).astype(int)
            df.sort_values('sort',inplace=True, ascending=True)
            df = df.drop('sort', axis=1)
            df.to_csv(os.path.join(config.eval_dir, "submission.csv"), index=False)
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        run_training(config)
    elif config.mode == "eval":
        run_eval(config)


if __name__ == '__main__':
    tf.app.run()
