from __future__ import division
from __future__ import print_function

import json
import os
import time
import numpy as np
import networkx as nx
from datetime import datetime

import logging
import scipy
from link_prediction import evaluate_link_prediction
from ParameterSetting import *
from models.DyMADC.models import DyMADC
from utils.minibatch import *
from utils.preprocess import *
from utils.utilities import *
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

np.random.seed(123)
tf.set_random_seed(123)

flags = tf.app.flags
FLAGS = flags.FLAGS

negative_examples_path = "DyMADC/data/"+FLAGS.dataset + "/" + FLAGS.dataset
output_dir = "DyMADC/logs/" + FLAGS.dataset
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

config_file = output_dir + "/flags_{}.json".format(FLAGS.dataset)

if (os.path.exists(config_file)):
    with open(config_file, 'r') as f:
        config = json.load(f)
        for name, value in config.items():
            if name in FLAGS.__flags:
                FLAGS.__flags[name].value = value

print("Updated flags", FLAGS.flag_values_dict().items())

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()

times='/{}_{}_{}'.format(today.year,today.month,today.day)
LOG_DIR = output_dir + FLAGS.log_dir + times
SAVE_DIR = output_dir + FLAGS.save_dir + times
Walk_DIR = "DyMADC/data/"+FLAGS.dataset + "/" 

def  create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

create_dir(LOG_DIR)
create_dir(SAVE_DIR)
create_dir(Walk_DIR)
log_file = LOG_DIR + '/%s_%s_%s_%s_%s.log' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))

log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

logging.info(FLAGS.flag_values_dict().items())
num_time_steps = FLAGS.time_steps
test_subset = FLAGS.test_subset
time_index = 2
FLAGS.dataname = split_data(test_subset, FLAGS.dataset+'.csv',FLAGS.dataset, split_nmuber = FLAGS.max_time-1, time_index = time_index, model_type="by_edgesNumber")

graphs, adjs, edges_graphs, test_edges ,train_edges, data= load_graphs(FLAGS.dataset,FLAGS.dataname)
if FLAGS.featureless:
    adjs = list(adjs)
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
else:
    feats = load_feats(FLAGS.dataset)

num_features = feats[0].shape[1]
assert num_time_steps < len(adjs) + 1

adj_train = []
feats_train = []
num_features_nonzero = []
context_pairs_train = get_context_pairs(num_time_steps, Walk_DIR)
train_edges, train_edges_false, test_edges, test_edges_false = divided_data(train_edges, test_edges, data, negative_examples_path)
new_G = nx.MultiGraph()
new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))

for e in graphs[num_time_steps - 2].edges():
    new_G.add_edge(e[0], e[1])

graphs[num_time_steps - 1] = new_G
adjs[num_time_steps - 1] = nx.adjacency_matrix(new_G)

adj_train = []
for adj in adjs:
    adj_train.append(normalize_graph_gcn(adj))

if FLAGS.featureless:
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in feats if x.shape[0] <= feats[num_time_steps - 1].shape[0]]
num_features = feats[0].shape[1]

feats_train = []
for feat in feats:
    feats_train.append(preprocess_features(feat)[1])

num_features_nonzero = [x[1].shape[0] for x in feats_train]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

placeholders = construct_placeholders(num_time_steps,num_features)

minibatchIterator = NodeMinibatchIterator(graphs, feats_train, adj_train,
                                          placeholders, num_time_steps, batch_size=FLAGS.batch_size,
                                          context_pairs=context_pairs_train)

model = DyMADC(placeholders, num_features, num_features_nonzero, minibatchIterator.degs)
sess.run(tf.global_variables_initializer())

epochs_test_result = defaultdict(lambda: [])
epochs_embeds = None
epochs_attn_wts_all = []

def operator_hadamard(u, v):
    return np.multiply(np.array(u), np.array(v))
def operator_l1(u, v):
    return np.abs(np.array(u) - np.array(v))
def operator_l2(u, v):
    return (np.array(u) - np.array(v)) ** 2
def operator_avg(u, v):
    return (np.array(u) + np.array(v)) / 2.0

operators = [operator_hadamard]
operators_name = ['operator_hadamard']

for epoch in range(FLAGS.epochs):
    minibatchIterator.shuffle()
    epoch_loss = 0.0
    it = 0
    epoch_time = 0.0
    while not minibatchIterator.end():
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({placeholders['spatial_drop']: FLAGS.spatial_drop})
        feed_dict.update({placeholders['temporal_drop']: FLAGS.temporal_drop})
        t = time.time()
        _, train_cost, graph_cost, reg_cost = sess.run([model.opt_op, model.loss, model.graph_loss, model.reg_loss],feed_dict=feed_dict) 
        epoch_time += time.time() - t
        logging.info("Mini batch Iter: {} train_loss= {:.5f} graph_loss= {:.5f} reg_loss= {:.5f}".format(it, train_cost, graph_cost, reg_cost))
        epoch_loss += train_cost
        it += 1

    logging.info("Time for epoch : {}".format(epoch_time))
    if (epoch + 1) % FLAGS.test_freq == 0:
        minibatchIterator.test_reset()
        emb = []
        feed_dict.update({placeholders['spatial_drop']: 0.0})
        feed_dict.update({placeholders['temporal_drop']: 0.0})
        if FLAGS.ttl < 0:
            assert FLAGS.time_steps == model.final_output_embeddings.get_shape()[1]
        emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:,model.final_output_embeddings.get_shape()[1] - 2, :]
        emb = np.array(emb)
        test_results, _ = evaluate_link_prediction(train_edges, train_edges_false, test_edges,test_edges_false, emb, emb
                                                              , operators
                                                              , operators_name
                                                              , minibatchIterator.index_mapping_nodeID[FLAGS.time_steps-1])

        test_max_score=0
        test_name = ''
        for k in test_results.keys():
            if(test_results.get(k)[0]>test_max_score):
                test_name = k
                test_max_score=test_results.get(k)[0]
        
        logging.info("Test results at epoch {}: AUC: {} best is : {} {}".format(epoch, test_results, test_name, test_max_score))
        if(len(operators)>1):
            operators=[operators[operators_name.index(test_name)]] #由第一次选出u v节点嵌入表示边的最好方法
            operators_name=[test_name]

        epoch_auc_test = test_results[test_name][1]

        epochs_test_result["max"].append(epoch_auc_test)
        if(max(epochs_test_result["max"])==test_max_score):
                epochs_embeds=emb
    epoch_loss /= it

best_epoch = epochs_test_result["max"].index(max(epochs_test_result["max"]))

print("Best epoch ", best_epoch)
logging.info("Best epoch {}".format(best_epoch))

test_results, _ = evaluate_link_prediction(train_edges, train_edges_false, test_edges, test_edges_false, epochs_embeds, epochs_embeds
                                                      , operators
                                                      , operators_name
                                                      , minibatchIterator.index_mapping_nodeID[FLAGS.time_steps-1])

print("Best epoch test results {}\n".format(test_results))
logging.info("Best epoch test results {}\n".format(test_results))
emb = epochs_embeds
np.savetxt(SAVE_DIR + '/embs_{}_{}.gz'.format(FLAGS.dataset, FLAGS.time_steps - 2), emb)
