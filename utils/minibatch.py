from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def construct_placeholders(num_time_steps,num_features):
    min_t = 0
    if FLAGS.ttl > 0:
        min_t = max(num_time_steps - FLAGS.ttl - 1, 0)
    placeholders = {
        'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, num_time_steps)],
        'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, num_time_steps)],
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),
        'features': [tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, num_time_steps)],
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
                 range(min_t, num_time_steps)],
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop')
    }
    return placeholders

class NodeMinibatchIterator(object):
    def __init__(self, graphs, features, adjs, placeholders, num_time_steps, context_pairs=None, batch_size=100):

        self.graphs = graphs
        self.features = features
        self.adjs = adjs
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.batch_num = 0
        self.num_time_steps = num_time_steps
        self.degs,self.index_mapping_nodeID = self.construct_degs() # yyh add. 
        self.context_pairs = context_pairs
        self.max_positive = FLAGS.neg_sample_size
        self.train_nodes = self.graphs[num_time_steps-1].nodes() # all nodes in the graph.
        print ("# train nodes", len(self.train_nodes))

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        degs = []
        index_mapping_nodeID = []
        for i in range(0, self.num_time_steps):
            G = self.graphs[i]
            deg = np.zeros((len(G.nodes()),))
            mapping_nodeId = np.zeros((len(G.nodes()),))
            for id,nodeid in enumerate(G.nodes()):
                neighbors = np.array(list(G.neighbors(nodeid)))
                deg[id] = len(neighbors)
                mapping_nodeId[id] = nodeid
            degs.append(deg)
            index_mapping_nodeID.append(mapping_nodeId) #记录nodeid映射位置
        min_t = 0
        if FLAGS.ttl > 0:
            min_t = max(self.num_time_steps - FLAGS.ttl - 1, 0)
        return degs[min_t:],index_mapping_nodeID

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes):
        """ Feed dict with (a) node pairs, (b) list of attribute matrices (c) list of snapshot adjs and metadata"""
        node_1_all = []
        node_2_all = []
        min_t = 0
        if FLAGS.ttl > 0:
            # For self-attention over a ttl of ``FLAGS.ttl'' time steps.
            min_t = max(self.num_time_steps - FLAGS.ttl - 1, 0)
        for t in range(min_t, self.num_time_steps):
            node_1 = []
            node_2 = []

            for n in batch_nodes: #context_pairs中对应节点n，选择《=max_positive（默认10）的邻居个数
                if len(self.context_pairs[t][n]) > self.max_positive:
                    node_1.extend([n]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][n], self.max_positive, replace=False))
                else:
                    node_1.extend([n]* len(self.context_pairs[t][n]))
                    node_2.extend(self.context_pairs[t][n])

            assert len(node_1) == len(node_2)
            assert len(node_1) <= self.batch_size * self.max_positive

            feed_dict = dict()
            node_1_all.append(node_1)
            node_2_all.append(node_2)

        self.adjs = list(self.adjs) #python 3

        feed_dict.update({self.placeholders['node_1'][t-min_t]:node_1_all[t-min_t] for t in range(min_t, self.num_time_steps)})#把新字典的键/值对更新到feed_dict里，张量
        feed_dict.update({self.placeholders['node_2'][t-min_t]:node_2_all[t-min_t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['features'][t-min_t]: self.features[t] for t in range(min_t, self.num_time_steps)}) #here has problem
        feed_dict.update({self.placeholders['adjs'][t-min_t]: self.adjs[t] for t in range(min_t, self.num_time_steps)})

        feed_dict.update({self.placeholders['batch_nodes']: np.array(batch_nodes).astype(np.int32)})
        return feed_dict

    def num_training_batches(self):
        """ Compute the number of training batches (using batch size)"""
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        """ Return the feed_dict for the next minibatch (in the current epoch) with random shuffling"""
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        """ Reset batch number"""
        self.train_nodes =  self.graphs[self.num_time_steps-1].nodes()
        self.batch_num = 0
