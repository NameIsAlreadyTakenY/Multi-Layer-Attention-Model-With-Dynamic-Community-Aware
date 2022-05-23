from __future__ import print_function
from math import floor
# from networkx.classes.function import edges
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
import dill
import os

def split_train_and_test_data(filename, test_subset=0.2):
    data = pd.read_csv(filename, usecols=["source", "target", "time"], header=0)
    n = floor(data.shape[0]*(1-test_subset))
    train = data[0:n]
    test = data[n:]
    test.index = range(0,len(test))
    return train,test


def data_split_by_date(test_subset, filename, time_index, split_nmuber):
    train,test = split_train_and_test_data(filename, test_subset=test_subset)
    start_time = train.iloc[0,2]
    end_time = train.iloc[-1,2]

    split_time = (end_time - start_time) // split_nmuber + 1

    edges = []
    re = []
    re_edges = []
    G = nx.MultiGraph()

    flag = start_time + split_time
    for line in train.values:
        u = int(line[0])
        v = int(line[1])
        time = float(line[time_index])

        if (time > flag):
            new_edges = edges.copy()
            new_G = nx.MultiGraph()
            if(len(G.edges()) > 0):
                G.remove_edges_from(G.edges())
            new_G.add_nodes_from(G.nodes())
            G.add_edges_from(edges)
            new_G.add_edges_from(edges)

            re.append(new_G)
            re_edges.append(new_edges)
            edges.clear()
            flag += split_time

            edges.append((u, v, {'time': time}))

    if(len(G.edges()) > 0):
        G.remove_edges_from(G.edges())
    G.add_edges_from(edges)
    re.append(G)
    re_edges.append(edges)

    temp_edges = []
    for x in test.values: #处理test数据
        source = int(x[0])
        target = int(x[1])
        temp_edges.append(([source,target,{'time': x[time_index]}]))

    test_graph = nx.MultiGraph()
    test_graph.add_nodes_from(G.nodes())
    test_graph.add_edges_from(temp_edges)
    re.append(test_graph)
    re_edges.append(temp_edges)
    
    return re, re_edges, train, test


def data_split_by_edgesNumber(test_subset, filename, time_index, split_nmuber):
    train,test = split_train_and_test_data(filename, test_subset=test_subset)
    line_count = train.shape[0]

    re = []
    re_edges = []
    n = line_count//split_nmuber

    G = nx.MultiGraph()
    for i in range(split_nmuber):
        if(len(G.edges()) > 0):
            G.remove_edges_from(G.edges())
        new_G = nx.MultiGraph()
        if(i < split_nmuber-1):
            split_data = train[i*n:(i+1)*n]
        else:
            split_data = train[i*n:]
        temp_edges = []
        for x in split_data.values:
            source = int(x[0])
            target = int(x[1])
            temp_edges.append(([source,target,{'time': x[time_index]}]))

        new_G.add_nodes_from(G.nodes())
        new_G.add_edges_from(temp_edges)
        G.add_edges_from(temp_edges)

        re.append(new_G)
        re_edges.append(temp_edges.copy())

    temp_edges = []
    for x in test.values:
        source = int(x[0])
        target = int(x[1])
        temp_edges.append(([source,target,{'time': x[time_index]}]))

    test_graph = nx.MultiGraph()
    test_graph.add_nodes_from(G.nodes())
    test_graph.add_edges_from(temp_edges)
    re.append(test_graph)
    re_edges.append(temp_edges.copy())
    return re, re_edges, train, test


def data_split_by_edgesNumber(test_subset, filename, time_index, split_nmuber, tot=1):
    train,test = split_train_and_test_data(filename, test_subset=test_subset)
    line_count = train.shape[0]

    re = []
    re_edges = []
    n = line_count//split_nmuber

    G = nx.MultiGraph()
    for i in range(split_nmuber):
        if(len(G.edges()) > 0):
            G.remove_edges_from(G.edges())  # 切换到tf_1环境执行
        new_G = nx.MultiGraph()
        if(i < split_nmuber-1):
            start_line = i-tot if i-tot > 0 else 0
            split_data = train[start_line*n:(i+1)*n]
        else:
            start_line = i-tot if i-tot>0 else 0
            split_data = train[start_line*n:]
        temp_edges = []
        for x in split_data.values:
            source = int(x[0])
            target = int(x[1])
            temp_edges.append(([source,target,{'time': x[time_index]}]))

        new_G.add_nodes_from(G.nodes())
        new_G.add_edges_from(temp_edges)
        G.add_edges_from(temp_edges)

        re.append(new_G)
        re_edges.append(temp_edges.copy())

    temp_edges = []
    for x in test.values:
        source = int(x[0])
        target = int(x[1])
        temp_edges.append(([source,target,{'time': x[time_index]}]))

    test_graph = nx.MultiGraph()
    test_graph.add_nodes_from(G.nodes())
    test_graph.add_edges_from(temp_edges)
    re.append(test_graph)
    re_edges.append(temp_edges.copy())
    return re, re_edges, train, test

def split_data(test_subset, dataset_name, dataset_dict_name, split_nmuber=4, time_index=2, model_type="by_date"):
    print(f"{dataset_name}  --  {dataset_dict_name}")
    dataset_path = f"Dataset/{dataset_dict_name}/{dataset_name}"
    output_dir = f"DyMADC/data/{dataset_dict_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = f"{output_dir}/{dataset_dict_name}_{split_nmuber}_{model_type}_{test_subset}"
    dataname = f'{dataset_dict_name}_{split_nmuber}_{model_type}_{test_subset}'
    if(os.path.exists(save_path+'.npz') and os.path.exists(save_path+'.pkl') and os.path.exists(save_path+'_train.pkl') and os.path.exists(save_path+'_test.pkl')):
        return dataname

    print(f'{dataset_name}  --  {dataset_dict_name} -- split_nmuber = {split_nmuber} -- model_type={model_type}-- test_subset={test_subset}')
    if(model_type == "by_date"):
        graphs, edges, train, test = data_split_by_date(test_subset, dataset_path, time_index=time_index, split_nmuber=split_nmuber)
    else:
        graphs, edges, train, test = data_split_by_edgesNumber(test_subset, dataset_path, time_index=time_index, split_nmuber=split_nmuber)
        # graphs, edges, train, test = data_split_by_edgesNumber(test_subset, dataset_path, time_index=time_index, split_nmuber=split_nmuber, tot=2)

    np.savez(save_path+'.npz', graph=graphs)
    dill.dump(edges, open(save_path+'.pkl', 'wb'))
    dill.dump(train, open(save_path+'_train.pkl', 'wb'))
    dill.dump(test, open(save_path+'_test.pkl', 'wb'))
    return dataname

def to_one_hot(labels, N, multilabel=False):
    ids, labels = zip(*labels)
    lb = MultiLabelBinarizer()
    if not multilabel:
        labels = [[x] for x in labels]
    lbs = lb.fit_transform(labels)
    encoded = np.zeros((N, lbs.shape[1]))
    for i in range(len(ids)):
        encoded[ids[i]] = lbs[i]
    return encoded

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
