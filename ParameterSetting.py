import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# General params for experiment setup - which need to be provided.
flags.DEFINE_float('test_subset', 0.25, 'test_subset')
flags.DEFINE_integer('max_time', 9, 'max_time step')
flags.DEFINE_string('dataset', 'fb-forum', 'Dataset string.')
flags.DEFINE_string('dataname', 'fb-forum_8_by_number.npz', 'dataset filename string.')
flags.DEFINE_integer('time_steps', 9, '# time steps to train (+1)')
flags.DEFINE_integer('GPU_ID', 0, 'GPU_ID')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('ttl', -1, '-1 => full)')
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_boolean('featureless', True, '1')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')

# Evaluation settings.
flags.DEFINE_integer('test_freq', 1, 'Testing frequency')
flags.DEFINE_integer('val_freq', 1, 'Validation frequency')

# Tunable hyper-parameters.
flags.DEFINE_integer('neg_sample_size', 10, 'give up')
flags.DEFINE_float('neg_weight', 1, 'pos:neg=1:1')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for self-attention model.')
flags.DEFINE_float('spatial_drop', 0.1, 'attn Dropout (1 - keep probability).')
flags.DEFINE_float('temporal_drop', 0.5, 'ffd Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_boolean('use_residual', False, 'Residual connections')

# Architecture configuration parameters.
flags.DEFINE_string('structural_head_config', '16', 'attention heads in each GAT layer')
flags.DEFINE_string('structural_layer_config', '128', 'GAT layer')

flags.DEFINE_string('temporal_head_config', '16', 'attention heads in each GAT layer')
flags.DEFINE_string('temporal_layer_config', '128', ' GAT layer')
flags.DEFINE_boolean('position_ffn', True, 'Use position wise feedforward')

# Generally static parameters -> Will not be updated by the argparse parameters.
flags.DEFINE_string('optimizer', 'adam', 'or SDG')
flags.DEFINE_integer('seed', 7, 'Random seed')

# Directory structure.
flags.DEFINE_string('save_dir', "/output", 'Save dir defaults to output/ within the base directory')
flags.DEFINE_string('log_dir', "/log", 'Log dir defaults to log/ within the base directory')