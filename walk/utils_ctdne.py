#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:43:20 2020

@author: chamezos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:23:32 2020

@author: Chamezopoulos Savvas
"""
import os
import time
import networkx as nx
import numpy as np
from stellargraph.data import TemporalRandomWalk, BiasedRandomWalk
from gensim.models import Word2Vec
import multiprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.metrics import recall_score, precision_score

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# Produces the node embeddings to be used by the ml
def ctdne_embedding(graph, name, output_dir, dataset_dict_name):

    num_walks_per_node = 10
    walk_length = 80
    context_window_size = 10
    dimensions = 128

    num_cw = len(graph.nodes()) * num_walks_per_node * (walk_length - context_window_size + 1)

    # save / load temporal_walks result
    if(os.path.exists(f"{output_dir}{name}_temporal_walks_{dataset_dict_name}.npy")):
        temporal_walks_result=np.load(f"{output_dir}{name}_temporal_walks_{dataset_dict_name}.npy",allow_pickle=True)
        temporal_walks_result=temporal_walks_result.tolist()

        print(f"{name}_temporal_walks_{dataset_dict_name}.npy load success")
    else:
        temporal_rw = TemporalRandomWalk(graph)
        temporal_walks_result = temporal_rw.run(num_cw=num_cw, cw_size=context_window_size, max_walk_length=walk_length, walk_bias="exponential")

        np.save(f"{output_dir}{name}_temporal_walks_{dataset_dict_name}.npy", temporal_walks_result)
        print(f"{name}_temporal_walks_{dataset_dict_name}.npy save success")

    print(f"Number of temporal random walks for '{name}': {len(temporal_walks_result)}")

    temporal_walks=[]
    for walks in temporal_walks_result:
        each_line=list(map(lambda walk: str(walk),walks))
        temporal_walks.append(each_line)

    temporal_model = Word2Vec(temporal_walks, size=dimensions, window=context_window_size, min_count=1, sg=1, workers=multiprocessing.cpu_count(), iter=1)
    temporal_model.wv.save_word2vec_format(f"{output_dir}{name}_temporal_model_{dataset_dict_name}.emb")
    print(f"temporal_model save success")

    def get_embedding(u):
        try:
            return temporal_model.wv[u]
        except KeyError:
            return np.zeros(dimensions)

    return get_embedding
    #return temporal_model

def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [binary_operator(transform_node(src), transform_node(dst)) for src, dst in link_examples]

def train_link_prediction_model(link_examples, link_labels, get_embedding, binary_operator):

    clf = link_prediction_classifier()
    link_features = link_examples_to_features(link_examples, get_embedding, binary_operator)

    clf.fit(link_features, link_labels)
    return clf

def link_prediction_classifier(max_iter=2000):

    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)

    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

def evaluate_link_prediction_model(clf, link_examples_test, link_labels_test, get_embedding, binary_operator):
    link_features_test = link_examples_to_features(link_examples_test, get_embedding, binary_operator)
    score_roc, score_acc, score_f1, score_rec, score_pres = evaluate_model(clf, link_features_test, link_labels_test)
    return score_roc, score_acc, score_f1, score_rec, score_pres


# applies the classifier on the test data and returns the auc score
def evaluate_model(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    auc = roc_auc_score(link_labels, predicted[:, positive_column])
    predicted = clf.predict(link_features)
    fpr, tpr, _ = roc_curve(link_labels, predicted)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    acc = accuracy_score(link_labels, predicted)
    f1 = f1_score(link_labels, predicted)
    rec = recall_score(link_labels, predicted)
    pres = precision_score(link_labels, predicted)

    return auc, acc, f1, rec, pres


def run_link_prediction(binary_operator, examples_train, labels_train, embedding_train, examples_model_selection, labels_model_selection):
    clf = train_link_prediction_model(examples_train, labels_train, embedding_train, binary_operator)
    score_auc, score_acc, score_f1, score_rec, score_pres = evaluate_link_prediction_model(clf, examples_model_selection, labels_model_selection, embedding_train, binary_operator)
    return {"classifier": clf, "binary_operator": binary_operator, "auc_score": score_auc, "acc_score": score_acc, "f1_score": score_f1, "precision_score": score_pres, "recall_score": score_rec}

def operator_hadamard(u, v):
    return u * v
def operator_l1(u, v):
    return np.abs(u - v)
def operator_l2(u, v):
    return (u - v)**2
def operator_avg(u, v):
    return (u + v) / 2.0

def node2vec_embedding(graph, name):
    p = 1.0
    q = 1.0
    dimensions = 128
    num_walks = 10
    walk_length = 80
    window_size = 10
    num_iter = 1
    workers = multiprocessing.cpu_count()

    rw = BiasedRandomWalk(graph)
    biased_walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)

    print(f"Number of random biased_walks for '{name}': {len(biased_walks)}")

    walks=[]
    for walks in biased_walks:
        each_line=list(map(lambda walk: str(walk),walks))
        walks.append(each_line)

    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=num_iter)
    def get_embedding(u):
        return model.wv[u]
    return get_embedding

def load_data(filename, weighted=False):
    if weighted == False:
        #validate filename and read graph
        try:
            G = nx.read_edgelist(filename, delimiter=',')

        except:
            print("Please enter a valid filename unweighted")
            return 0
        edgelist_df = nx.to_pandas_edgelist(G)
        edgelist_df.columns = ['source', 'target']
    else:
        #validate filename and read graph
        try:
            # G = nx.read_weighted_edgelist(filename, delimiter=',')
            G = nx.MultiDiGraph()
            edges = []
            with open(filename) as f:
                for line in f:
                    tokens = line.replace("\n", "").split(',')
                    u = int(tokens[0])
                    v = int(tokens[1])
                    time = float(tokens[2])
                    edges.append((u, v, {'time': time}))
            G.add_edges_from(edges)
        except:
            print("Please enter a valid filename")
            return 0
        edgelist_df = nx.to_pandas_edgelist(G)
        edgelist_df.columns = ['source', 'target', 'time']
    return G, edgelist_df

def data_load(filename, time_index):
    try:
        G = nx.MultiDiGraph()
        edges = []
        with open(filename) as f:
            for line in f:
                tokens = line.strip().split()
                u = int(tokens[0])
                v = int(tokens[1])
                time = float(tokens[time_index])
                edges.append((u, v, {'time': time}))
        G.add_edges_from(edges)
    except:
        print("Please enter a valid filename")
        return 0
    edgelist_df = nx.to_pandas_edgelist(G)
    edgelist_df.columns = ['source', 'target', 'time']
    return G, edgelist_df
