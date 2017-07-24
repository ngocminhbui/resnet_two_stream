import numpy as np
from resnet_architecture import *
import tensorflow as tf
import os
from datetime import datetime
from exp_config import *
from input import distorted_inputs,inputs

FLAGS = tf.app.flags.FLAGS

def top_k_error(predictions, labels, k):
    batch_size =float(FLAGS.batch_size) # predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size
def num_correct(predictions,labels,k):
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    return tf.reduce_sum(in_top1)

def evaluate(is_training, logits, images, labels):
    predictions = tf.nn.softmax(logits)
    top1_error = top_k_error(predictions, labels, 1)
    top1_num_correct = num_correct(predictions,labels,1)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    summary_op = tf.summary.merge_all()

    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)
    tf.train.start_queue_runners(sess=sess)

    # restore checkpoint
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        except Exception:
            return
    else:
        print 'No checkpoints found'
        return

    # evaluation
    coord = tf.train.Coordinator()
    data = None
    nCorrect=0.0
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        num_iter = int(np.ceil(NUM_EXAMPLES / FLAGS.batch_size))
        # true_count = 0-
        precision = 0
        # total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            top1_error_value,top1_num_correct_value,scores_, lbls_ = sess.run([top1_error,top1_num_correct, predictions, labels], {is_training: False})
            preds_ = scores_.argmax(axis=1)
            preds_ = np.expand_dims(preds_, axis=1)
            lbls_ = np.expand_dims(lbls_, axis=1)
            tmp = np.concatenate((scores_, preds_, lbls_), axis=1)
            if data is None:
                data = tmp
            else:
                data = np.concatenate((data, tmp), axis=0)
            # _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
            precision += 1 - top1_error_value
            nCorrect+=top1_num_correct_value
            # print('Validation top1 error %.2f' % top1_error_value)
            step += 1

        # precision
        precision /= num_iter
        print '%s : precision = %.8f' % (datetime.now(), precision)
        print '%s : precision = %.8f' % (datetime.now(),nCorrect/NUM_EXAMPLES)
        # write summary
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op, {is_training: False}))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)
    except Exception as e:
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    # write results
    fn = os.path.join(FLAGS.eval_dir, 'score_' + str(global_step) + '.txt')
    np.savetxt(fn, data, fmt='%.4f')

    return


def main(_):
    global NUM_EXAMPLES
    NUM_EXAMPLES = len(open(FLAGS.eval_lst, 'r').read().splitlines())


    images, labels = inputs(FLAGS.data_dir, FLAGS.eval_lst)

    is_training = tf.placeholder('bool', [], name='is_training')  # placeholder for the fusion part

    logits = inference(images,
                       num_classes=FLAGS.num_classes,
                       is_training=is_training,
                       num_blocks=[3, 4, 6, 3])
    evaluate(is_training,logits, images, labels)
    return

if __name__ == '__main__':
    tf.app.run(main)
