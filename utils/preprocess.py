from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
import dill
import os
import random
import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(123)

def load_graphs(dataset_str,dataset_name):
    path = f"DyMADC/data/{dataset_str}/{dataset_name}"
    dataset_path = f"Dataset/{dataset_str}/{dataset_str}.csv"
    data = pd.read_csv(dataset_path, usecols=["source", "target", "time"], header=0)
    if(os.path.exists(path+'.npz') and os.path.exists(path+'.pkl')):
        graphs = np.load(path+'.npz', allow_pickle=True, encoding="latin1")['graph']
        print("Loaded {} graphs ".format(len(graphs)))
        adj_matrices=[]
        for graph in graphs:
            adj_matrices.append(nx.adjacency_matrix(graph))
        with open(path+'.pkl', 'rb') as f:
            edges = dill.load(f)
            f.close()
        with open(path+'_test.pkl', 'rb') as f:
            test_edges = dill.load(f)
            f.close()
        with open(path+'_train.pkl', 'rb') as f:
            train_edges = dill.load(f)
            f.close()
    else:
        raise ValueError('graphs data is None')
    return graphs,adj_matrices,edges,test_edges,train_edges, data


def load_feats(dataset_str):
    features = np.load("DyMADC/data/{}/{}".format(dataset_str, "features.npz"), allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_graph_gcn(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def get_context_pairs(num_time_steps, Walk_DIR):
    filename = "/train_pairs_CDWalk_{}.pkl".format(str(num_time_steps - 2))
    path = Walk_DIR + filename
    try:
        context_pairs_train = dill.load(open(path, 'rb'))
        print("Loaded success")
        return context_pairs_train
    except (IOError, EOFError):
        print("Loaded data from file directly failed")

def positive_and_negative_links(g, edges, negative_examples_path):
        pos = list(edges[["source", "target"]].itertuples(index=False))
        neg = sample_negative_examples(g, pos, negative_examples_path, edges)
        return pos, neg

def sample_negative_examples(g, positive_examples, negative_examples_path, edges):
    positive_examples_number = len(positive_examples)
    def valid_neg_edge(src, tgt):
        return (
            src != tgt and
            (src, tgt) not in positive_examples and (tgt, src) not in positive_examples)

    try:
        possible_neg_edges = dill.load(open(f'{negative_examples_path}_negative_examples.pkl', 'rb'))
    except (IOError, EOFError):
        if(len(g.nodes()) * len(g.nodes()) > int(edges.shape[0])):
            possible_neg_edges = []
            graph_nodes = g.nodes()
            nodes_number = len(graph_nodes)
            while len(possible_neg_edges) < 2 * positive_examples_number:
                id_src=np.random.randint(0,nodes_number)
                id_tgt=np.random.randint(0,nodes_number)
                if valid_neg_edge(graph_nodes[id_src], graph_nodes[id_tgt]):
                    possible_neg_edges.append((graph_nodes[id_src],graph_nodes[id_tgt]))
        else:
            possible_neg_edges = [(src, tgt) for src in g.nodes() for tgt in g.nodes() if valid_neg_edge(src, tgt)]
        dill.dump(possible_neg_edges, open(f'{negative_examples_path}_negative_examples.pkl', 'wb'))
    if(len(possible_neg_edges)>=positive_examples_number):
        return random.sample(possible_neg_edges, k=positive_examples_number)
    else:
        possible_neg_edges = possible_neg_edges*int((positive_examples_number/len(possible_neg_edges))+1)
        return random.sample(possible_neg_edges, k=positive_examples_number)

def divided_data(train_edges, test_edges, data, negative_examples_path):
    temp_edges = []
    for x in data.values:
        source = int(x[0])
        target = int(x[1])
        temp_edges.append(([source,target]))

    graph = nx.Graph()
    graph.add_edges_from(temp_edges)

    train_temp_edges = []
    for x in test_edges.values:
        source = int(x[0])
        target = int(x[1])
        train_temp_edges.append(([source,target]))

    train_graph = nx.Graph()
    train_graph.add_edges_from(temp_edges)
    pos_train, neg_train = positive_and_negative_links(graph, train_edges, negative_examples_path)
    pos_test, neg_test = positive_and_negative_links(graph, test_edges, negative_examples_path)
    return list(pos_train), list(neg_train), list(pos_test), list(neg_test)