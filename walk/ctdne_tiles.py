from collections import defaultdict
import networkx as nx
import dill
import pandas as pd
from sklearn.model_selection import train_test_split
from CD_temporal_walk import CD_temporal_walk
# import TILES_time as t
import TILES_number as t

dataset_names = ["fb-forum.csv"]
dataset_dict_names = ["fb-forum"]

assert(len(dataset_names)==len(dataset_dict_names))

if __name__ == '__main__':
    for i in range(len(dataset_names)):
        total_result=[]
        dataset_name = dataset_names[i]
        dataset_dict_name = dataset_dict_names[i]
        dataset_path = "Dataset/" + dataset_dict_name + '/'+dataset_name
        output_dir = "DyMADC/data/" + dataset_dict_name
        split_nmuber = 8
        filename = "/train_pairs_CDWalk_{}.pkl".format(str(split_nmuber - 1))
        pkl_path = output_dir + filename
        train_subset = 0.75
        test_subset = 0.25
        num_walks_per_node = 10
        walk_length = 80
        WINDOW_SIZE = 5
        #-------------------------------------------导入数据，生成图，切分训练，测试边,正负采样--------------------------------------------------
        data= pd.read_csv(dataset_path, usecols=["source","target","time"] ,header=0)
        edges=pd.DataFrame(data)

        num_edges_graph = int(len(edges) * (train_subset))

        edges_graph = edges[:num_edges_graph]
        edges_other = edges[num_edges_graph:]

        edges_train, edges_test = train_test_split(edges_other, test_size=test_subset)
        n = len(edges_graph)//split_nmuber
        obs=[]
        for i in range(split_nmuber):
            obs.append(data.iat[n*i, -1])
        obs.append(data.iat[-1, -1])

        ls = [i+1 for i in range(split_nmuber-1)]
        et = t.TILES(data_df=edges_graph, obs=obs, tot=obs[0])
        nodes_com_d,com_nodes_d,graphs_edges=et.execute()
        for i in ls:
            graphs_edges[i] = pd.concat([graphs_edges[i]+graphs_edges[i-1]])


        NS = []
        jobs = []
        for j in range(len(nodes_com_d)):
            temp_edges = []
            node_time_index = defaultdict(lambda: [])
            for x in graphs_edges[j].values:
                source = int(x[0])
                target = int(x[1])
                time = x[2]
                node_time_index[source].append(time)
                node_time_index[target].append(time)
                temp_edges.append(tuple([source,target,time]))
            graph = nx.MultiGraph()
            graph.add_weighted_edges_from(temp_edges,weight='time')
            num_cw = len(graph.nodes()) * num_walks_per_node * (walk_length - WINDOW_SIZE + 1)
            temporal_rw = CD_temporal_walk(graph,graphs_edges[j],nodes_com_d=nodes_com_d[j],com_nodes_d=com_nodes_d[j],node_time_index=node_time_index)
            temporal_walks = temporal_rw.run(num_cw=num_cw,cw_size=WINDOW_SIZE,max_walk_length=walk_length,walk_bias="exponential")
            
            pairs = defaultdict(lambda: [])
            for walk in temporal_walks:
                for word_index, word in enumerate(walk):
                    for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                        if nb_word != word:
                            pairs[word].append(nb_word)
            NS.append(pairs)

        dill.dump(NS, open(pkl_path, 'wb'))