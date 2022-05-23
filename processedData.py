from __future__ import print_function
import numpy as np
import pandas as pd
import networkx as nx
import random
import dill
from utils.utilities import *

def positive_and_negative_links(g, edges, negative_examples_path):
        # pos = list(edges[["source", "target"]].drop_duplicates(subset=['source', 'target'], keep='first', inplace=False).itertuples(index=False))  # 去除动态网络中重边
        pos = list(edges[["source", "target"]].itertuples(index=False))
        neg = sample_negative_examples(g, pos, negative_examples_path, edges)
        return pos, neg

def sample_negative_examples(g, positive_examples, negative_examples_path, edges):
    positive_examples_number = len(positive_examples)
    positive_examples = list(edges[["source", "target"]].drop_duplicates(subset=['source', 'target'], keep='first', inplace=False).itertuples(index=False))
    def valid_neg_edge(src, tgt):
        return (
            # no self-loops
            src != tgt and
            (src, tgt) not in positive_examples and (tgt, src) not in positive_examples)

    try:
        print(0)
        possible_neg_edges = dill.load(open(f'{negative_examples_path}_negative_examples.pkl', 'rb'))
    except (IOError, EOFError):
        if(len(g.nodes()) * len(g.nodes()) > int(edges.shape[0])):
            print(1)
            possible_neg_edges = []
            graph_nodes = g.nodes()
            nodes_number = len(graph_nodes)
            while len(possible_neg_edges) < 2 * positive_examples_number:
                id_src=np.random.randint(0,nodes_number)
                id_tgt=np.random.randint(0,nodes_number)
                if valid_neg_edge(graph_nodes[id_src], graph_nodes[id_tgt]):
                    possible_neg_edges.append((graph_nodes[id_src],graph_nodes[id_tgt]))
        else:
            print(2)
            possible_neg_edges = [(src, tgt) for src in g.nodes() for tgt in g.nodes() if valid_neg_edge(src, tgt)]
        dill.dump(possible_neg_edges, open(f'{negative_examples_path}_negative_examples.pkl', 'wb'))
    if(len(possible_neg_edges)>=positive_examples_number):
        return random.sample(possible_neg_edges, k=positive_examples_number)
    else:
        possible_neg_edges = possible_neg_edges*int((positive_examples_number/len(possible_neg_edges))+1)
        return random.sample(possible_neg_edges, k=positive_examples_number)

# 输入测试边，返回训练，测试用例的正负采样
def get_evaluation_data_yyh(graphsnap, train_edges, test_edges, data, negative_examples_path, time_index=2):
    temp_edges = []
    for x in data.values: #dataframe格式数据转化为数组格式，train_edges生成graph
        source = int(x[0])
        target = int(x[1])
        # temp_edges.append(([source,target,{'time': x[time_index]}]))
        temp_edges.append(([source,target]))

    graph = nx.Graph()
    graph.add_edges_from(temp_edges)

    train_temp_edges = []
    for x in test_edges.values: #dataframe格式数据转化为数组格式
        source = int(x[0])
        target = int(x[1])
        train_temp_edges.append(([source,target]))

    train_graph = nx.Graph()
    train_graph.add_edges_from(temp_edges)
    print(f'{len(train_graph.nodes()), len(graph.nodes())}')
    pos_train, neg_train = positive_and_negative_links(graph, train_edges, negative_examples_path)
    pos_test, neg_test = positive_and_negative_links(graph, test_edges, negative_examples_path)
    if(len(train_graph.nodes())!=len(graph.nodes())):
        pos_test = delete_illegal_edge(train_graph.nodes(), pos_test)
        neg_test = delete_illegal_edge(train_graph.nodes(), neg_test)
    print("# train examples: ", len(pos_train), len(neg_train))
    print("# test examples:", len(pos_test), len(neg_test))
    return list(pos_train), list(neg_train), list(pos_test), list(neg_test)

def delete_illegal_edge(nodes, ori_edges):
    edges = []
    for e in ori_edges:
        if e[0] in nodes and e[1] in nodes:
            edges.append((e[0],e[1]))
    return edges
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


# datasets = ["fb-messages","wikipedia","reddit"]
datasets = ["fb-forum","contacts-prox-high-school-2013","ia-reality-call"]
for dataset in datasets:
    negative_examples_path = "DyMADC/data/"+dataset + "/" + dataset
    num_time_steps = 9
    test_subset = 0.25
    time_index = 2
    dataname = split_data(test_subset, dataset+'.csv',dataset, split_nmuber = 8, time_index = time_index, model_type="by_edgesNumber")
    graphs, adjs, edges_graphs, test_edges ,train_edges, data = load_graphs(dataset,dataname)
    train_edges, train_edges_false, test_edges, test_edges_false = get_evaluation_data_yyh(graphs[num_time_steps-1], train_edges, test_edges, data, negative_examples_path, time_index)


