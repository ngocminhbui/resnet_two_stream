from resnet_train import train
from resnet_architecture import *
from exp_config import *
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

''' Load list of  {filename, label_name, label_index} '''
def load_data(data_dir, data_lst):
    data = []
    train_lst = open(data_lst, 'r').read().splitlines()
    dictionary = open(FLAGS.dictionary, 'r').read().splitlines()
    for img_fn in train_lst:
        fn = os.path.join(data_dir, img_fn + '_crop.png')
        fn_depth = os.path.join(data_dir,img_fn + '_depthcrop.png')
        label_name = img_fn.split('/')[0]
        label_index = dictionary.index(label_name)
        data.append({
            "filename": fn,
            "depthname": fn_depth,
            "label_name": label_name,
            "label_index": label_index
        })
    return data


''' Load input data using queue (feeding)'''


def read_image_from_disk(input_queue):
    label = input_queue[2]


    file_contents = tf.read_file(input_queue[0])
    depth_contents = tf.read_file(input_queue[1])

    file_contents = tf.image.decode_png(file_contents, channels=3)
    file_contents=tf.cast(file_contents,tf.float32)

    depth_contents = tf.image.decode_png(depth_contents, channels=3)
    depth_contents = tf.cast(depth_contents, tf.float32)
    ''' Image Normalization (later...) '''


    return file_contents, depth_contents, label


def distorted_rgb_d_inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [d['filename'] for d in data]
    depthnames = [d['depthname'] for d in data]
    label_indexes = [d['label_index'] for d in data]

    input_queue = tf.train.slice_input_producer([filenames,depthnames, label_indexes], shuffle=True)

    # read image and label from disk
    image, depth, label = read_image_from_disk(input_queue)

    ''' Data Augmentation '''
    image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, 3])
    image = tf.image.random_flip_left_right(image)

    depth = tf.random_crop(depth, [FLAGS.input_size, FLAGS.input_size, 3])


    # generate batch
    image_batch,depth_batch, label_batch = tf.train.shuffle_batch(
        [image, depth, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples)

    return image_batch, depth_batch, tf.reshape(label_batch, [FLAGS.batch_size])


def main(_):
    images, depths, labels = distorted_rgb_d_inputs(FLAGS.data_dir, FLAGS.train_lst)

    is_training = tf.placeholder('bool', [], name='is_training')  # placeholder for the fusion part

    logits_fuse, logits_image, logits_depth = inference(images,depths,
                       num_classes=FLAGS.num_classes,
                       is_training=is_training,
                       num_blocks=[3, 4, 6, 3])
    train(is_training,[logits_fuse,logits_image,logits_depth], images, labels)


if __name__ == '__main__':
    tf.app.run(main)
