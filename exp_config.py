import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_lst', './lists/train_mini_1.lst', 'training list')
tf.app.flags.DEFINE_string('eval_lst', './lists/eval_mini_1.lst', 'validation list')
tf.app.flags.DEFINE_string('train_dir', './log/mini_1','Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_string('eval_dir', './log/eval_mini_1', 'save eval')
tf.app.flags.DEFINE_string('dictionary', './lists/dictionary.lst', 'dictionary')
#tf.app.flags.DEFINE_string('data_dir', '/home/knmac/data/rgbd-dataset-processed-4dpng','data dir')
tf.app.flags.DEFINE_string('data_dir', '/media/ngocminh/DATA/rgbd-dataset-processed-fringe','data dir')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning  rate.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_integer('max_steps', 500000, 'max steps')
#tf.app.flags.DEFINE_boolean('resume', True, 'resume from latest saved state')
tf.app.flags.DEFINE_boolean('resume', False, 'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,'produce fewer summaries to save HD space')
tf.app.flags.DEFINE_float('starter_learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('train_decay_rate', 0.1, 'decay rate of training phase')
tf.app.flags.DEFINE_integer('train_decay_steps', 10000, 'number of steps before decaying')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')

tf.app.flags.DEFINE_integer('num_classes', 10, 'number of classes')
tf.app.flags.DEFINE_integer('input_size', 224, 'width and height of image')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 16, 'number of preprocess threads')
tf.app.flags.DEFINE_integer('min_queue_examples', 20000, 'min after dequeue')

tf.app.flags.DEFINE_string('pretrained_model', './model/ResNet-L50.npy', "Path of resnet pretrained model")

