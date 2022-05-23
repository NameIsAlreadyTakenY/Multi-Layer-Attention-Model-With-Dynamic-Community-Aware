"""
    Created on 11/feb/2015
    @author: Giulio Rossetti
"""
from sys import path
import networkx as nx
import datetime
import time

from numpy.lib.utils import source
from future.utils import iteritems

from collections import defaultdict

import sys
if sys.version_info > (2, 7):
    from io import StringIO
    from queue import PriorityQueue
else:
    from cStringIO import StringIO
    from Queue import PriorityQueue


__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"


class TILES(object):
    def __init__(self, data_df=None, g=None, tot=float('inf'), obs=7, path="", start=0, end=0):
        self.path = path
        self.tot = tot
        self.cid = 0
        self.actual_slice = 0
        if g is None:
            self.g = nx.Graph()
        else:
            self.g = g
        self.splits = None
        self.spl = StringIO()
        self.status = open("%s/extraction_status.txt" % (path), "w")
        self.removed = 0
        self.added = 0
        self.start = start
        self.end = end
        self.obs = obs
        self.communities = {}

        self.nodes_with_communities = {}
        self.data = data_df

        self.graphs_edges=[]
        self.communities_nodes=[]
        self.nodes_communities=[]

        self.nodes_com_d=[]
        self.com_nodes_d=[]
        self.obs_index=1
    def execute(self):
        self.status.write(u"Started! (%s) \n\n" % str(time.asctime(time.localtime(time.time()))))
        self.status.flush()

        qr = PriorityQueue()

        actual_time=float(self.data.iat[0,-1])
        last_break = actual_time
        self.tot = actual_time

        count = 0

        #################################################
        #                   Main Cycle                  #
        #################################################
        for i in range(len(self.data)):

            self.added += 1
            e = {}
            u = int(self.data.iat[i,0])
            v = int(self.data.iat[i,1])
            dt = float(self.data.iat[i,2])

            e['weight'] = 1  #这里用来标记边为几重边，可以根据Hawkes process公式计算权重
            e["u"] = u
            e["v"] = v
            e["time"] = dt

            ##todo 改写时间
            #############################################
            #               Observations                #
            #############################################

            # gap = dt - last_break

            if dt >= self.obs[self.obs_index]:  # 达到时间窗口，输出结果
                last_break = dt
                self.obs_index += 1 #更新窗口时间点
                if(self.obs_index>1):
                    self.tot=self.obs[self.obs_index-2]
                self.added -= 1

                print("New slice. Starting time: %s" % dt)

                self.status.write(u"Saving Slice %s: Starting %s ending %s - (%s)\n" %
                                  (self.actual_slice, actual_time, dt,
                                   str(time.asctime(time.localtime(time.time())))))

                self.status.write(u"Edge Added: %d\tEdge removed: %d\n" % (self.added, self.removed))
                self.added = 1
                self.removed = 0

                actual_time = dt
                self.status.flush()

                self.spl = StringIO()

                self.print_communities(i)
                self.start = i
                self.status.write(u"\nStarted Slice %s (%s)\n" % (self.actual_slice, str(datetime.datetime.now().time())))

            if u == v:
                continue

            if self.tot != float('inf'):#float('inf')表示正无穷
                qr.put((dt, (int(e['u']), int(e['v']), int(e['weight']))))#qr优先队列，按照dt来排队，dt最小的值先出队
                self.remove(dt, qr)# 检查新边出现的时间点是否有旧的边需要remove

            if not self.g.has_node(u):
                self.g.add_node(u)
                self.g.node[u]['c_coms'] = {}  # central

            if not self.g.has_node(v):
                self.g.add_node(v)
                self.g.node[v]['c_coms'] = {}

            if self.g.has_edge(u, v):  # 重边的话，权重叠加
                w = self.g.adj[u][v]["weight"]
                self.g.adj[u][v]["weight"] = w + e['weight']
                continue
            else:
                self.g.add_edge(u, v)
                self.g.adj[u][v]["weight"] = e['weight']

            u_n = list(self.g.neighbors(u))
            v_n = list(self.g.neighbors(v))

            #############################################
            #               Evolution                   #
            #############################################

            # new community of peripheral nodes (new nodes)
            if len(u_n) > 1 and len(v_n) > 1:
                common_neighbors = set(u_n) & set(v_n)
                self.common_neighbors_analysis(u, v, common_neighbors)# 处理两个节点都已存在于网络中的一般情况，是否创建新社区，是否开始

            count += 1

        #  Last writing
        self.status.write(u"Slice %s: Starting %s ending %s - (%s)\n" %(self.actual_slice, actual_time, actual_time,
                           str(time.asctime(time.localtime(time.time())))))
        self.status.write(u"Edge Added: %d\tEdge removed: %d\n" % (self.added, self.removed))
        self.added = 0
        self.removed = 0

        self.print_communities(len(self.data))
        self.status.write(u"Finished! (%s)" % str(time.asctime(time.localtime(time.time()))))
        self.status.flush()
        self.status.close()

        return self.nodes_com_d,self.com_nodes_d,self.graphs_edges

    @property
    def new_community_id(self):
        """
            Return a new community identifier
            :return: new community id
        """
        self.cid += 1
        self.communities[self.cid] = {}
        return self.cid

    def remove(self, actual_time, qr):
        """
            Edge removal procedure
            :param actual_time: timestamp of the last inserted edge
            :param qr: Priority Queue containing the edges to be removed ordered by their timestamps
        """

        coms_to_change = {}
        at = actual_time

        # main cycle on the removal queue
        if not qr.empty():

            t = qr.get()
            timestamp = t[0]
            e = (t[1][0],  t[1][1], t[1][2])

            # delta = at - timestamp

            if timestamp >= self.tot:#如果优先队列中最前的边仍然在ttl内，将边再次入队列
                qr.put((timestamp, t[1]))

            else:
                while self.tot > timestamp:

                    self.removed += 1
                    u = int(e[0])
                    v = int(e[1])
                    if self.g.has_edge(u, v):

                        w = self.g.adj[u][v]["weight"]

                        # decreasing link weight if greater than one
                        # (multiple occurrence of the edge: remove only the oldest)
                        if w > 1:
                            self.g.adj[u][v]["weight"] = w - 1
                            e = (u,  v, w-1)
                            qr.put((at, e))

                        else:
                            # u and v shared communities
                            if len(list(self.g.neighbors(u))) > 1 and len(list(self.g.neighbors(v))) > 1:
                                coms = set(self.g.node[u]['c_coms'].keys()) & set(self.g.node[v]['c_coms'].keys())

                                for c in coms:
                                    if c not in coms_to_change:
                                        cn = set(self.g.neighbors(u)) & set(self.g.neighbors(v))
                                        coms_to_change[c] = [u, v]
                                        coms_to_change[c].extend(list(cn))
                                    else:
                                        cn = set(self.g.neighbors(u)) & set(self.g.neighbors(v))
                                        coms_to_change[c].extend(list(cn))
                                        coms_to_change[c].extend([u, v])
                                        ctc = set(coms_to_change[c])
                                        coms_to_change[c] = list(ctc)
                            else:
                                if len(list(self.g.neighbors(u))) < 2:
                                    coms_u = [x for x in self.g.node[u]['c_coms'].keys()]
                                    for cid in coms_u:
                                        self.remove_from_community(u, cid)

                                if len(list(self.g.neighbors(v))) < 2:
                                    coms_v = [x for x in self.g.node[v]['c_coms'].keys()]
                                    for cid in coms_v:
                                        self.remove_from_community(v, cid)

                            self.g.remove_edge(u, v)

                    if not qr.empty():
                        t = qr.get()
                        timestamp = t[0]
                        delta = at - timestamp
                        e = t[1]

        # update of shared communities
        self.update_shared_coms(coms_to_change)

    def update_shared_coms(self, coms_to_change):
        # update of shared communities
        for c in coms_to_change:
            if c not in self.communities:
                continue

            c_nodes = self.communities[c].keys()

            if len(c_nodes) > 3:

                sub_c = self.g.subgraph(c_nodes)
                c_components = nx.number_connected_components(sub_c)

                # unbroken community
                if c_components == 1:
                    to_mod = sub_c.subgraph(coms_to_change[c])
                    self.modify_after_removal(to_mod, c)

                # broken community: bigger one maintains the id, the others obtain a new one
                else:
                    new_ids = []

                    first = True
                    components = nx.connected_components(sub_c)
                    for com in components:
                        if first:
                            if len(com) < 3:
                                self.destroy_community(c)
                            else:
                                to_mod = list(
                                    set(com) & set(coms_to_change[c]))
                                sub_c = self.g.subgraph(to_mod)
                                self.modify_after_removal(sub_c, c)
                            first = False

                        else:
                            if len(com) > 3:
                                # update the memberships: remove the old ones and add the new one
                                to_mod = list(
                                    set(com) & set(coms_to_change[c]))
                                sub_c = self.g.subgraph(to_mod)

                                central = self.centrality_test(sub_c).keys()
                                if len(central) >= 3:
                                    actual_id = self.new_community_id
                                    new_ids.append(actual_id)
                                    for n in central:
                                        self.add_to_community(n, actual_id)

                    # splits
                    if len(new_ids) > 0 and self.actual_slice > 0:
                        self.spl.write(u"%s\t%s\n" % (c, str(new_ids)))
            else:
                self.destroy_community(c)

    def modify_after_removal(self, sub_c, c):
        """
            Maintain the clustering coefficient invariant after the edge removal phase
            :param sub_c: sub-community to evaluate
            :param c: community id
        """
        central = self.centrality_test(sub_c).keys()

        # in case of previous splits, update for the actual nodes
        remove_node = set(self.communities[c].keys()) - set(sub_c.nodes())

        for rm in remove_node:
            self.remove_from_community(rm, c)

        if len(central) < 3:
            self.destroy_community(c)
        else:
            not_central = set(sub_c.nodes()) - set(central)
            for n in not_central:
                self.remove_from_community(n, c)

    # 处理两个节点都已存在于网络中的一般情况，是否创建新社区，是否开始
    def common_neighbors_analysis(self, u, v, common_neighbors):
        """
            General case in which both the nodes are already present in the net.
            :param u: a node
            :param v: a node
            :param common_neighbors: common neighbors of the two nodes
        """
        # no shared neighbors
        if len(common_neighbors) < 1:
            return
        else:
            shared_coms = set(self.g.node[v]['c_coms'].keys()) & set(self.g.node[u]['c_coms'].keys())
            only_u = set(self.g.node[u]['c_coms'].keys()) - set(self.g.node[v]['c_coms'].keys())
            only_v = set(self.g.node[v]['c_coms'].keys()) - set(self.g.node[u]['c_coms'].keys())

            # community propagation: a community is propagated iff at least two of [u, v, z] are central
            propagated = False

            for z in common_neighbors:
                for c in self.g.node[z]['c_coms'].keys():
                    if c in only_v:
                        self.add_to_community(u, c)
                        propagated = True

                    if c in only_u:
                        self.add_to_community(v, c)
                        propagated = True

                for c in shared_coms:
                    if c not in self.g.node[z]['c_coms']:
                        self.add_to_community(z, c)
                        propagated = True

            else:
                if not propagated:
                    # new community
                    actual_cid = self.new_community_id
                    self.add_to_community(u, actual_cid)
                    self.add_to_community(v, actual_cid)

                    for z in common_neighbors:
                        self.add_to_community(z, actual_cid)

    # 输出划分的社区信息
    def print_communities(self,end_index):
        """
            Print the actual communities
        """

        nodes_to_coms = {}
        merge = {}
        coms_to_remove = []
        drop_c = []

        for idc, comk in iteritems(self.communities):
            com = comk.keys()
            if self.communities[idc] is not None:
                if len(com) > 2:
                    key = tuple(sorted(com))

                    # Collision check and merge index build (maintaining the lowest id)
                    if key not in nodes_to_coms:
                        nodes_to_coms[key] = idc
                    else:
                        old_id = nodes_to_coms[key]
                        drop = idc
                        if idc < old_id:
                            drop = old_id
                            nodes_to_coms[key] = idc

                        # merged to remove
                        coms_to_remove.append(drop)
                        if not nodes_to_coms[key] in merge:
                            merge[nodes_to_coms[key]] = [idc]
                        else:
                            merge[nodes_to_coms[key]].append(idc)
                else:
                    drop_c.append(idc)
            else:
                drop_c.append(idc)

        for dc in drop_c:
            self.destroy_community(dc)

        nodes_communities_dict=defaultdict(list)
        communities_nodes_dict=defaultdict(list)
            
        for k, idk in iteritems(nodes_to_coms):
            communities_nodes_dict[idk]=list(k)
            # self.nodes_with_communities处理
            for n in list(k):
                if n in self.nodes_with_communities:
                    self.nodes_with_communities[n][idk] = None
                else:
                    self.nodes_with_communities[n] = {idk: None}
            
        for k, idk in iteritems(self.nodes_with_communities):
            nodes_communities_dict[k]=list(idk.keys())

        self.nodes_com_d.append(nodes_communities_dict)
        self.com_nodes_d.append(communities_nodes_dict)
        self.graphs_edges.append(self.data.iloc[self.start:end_index])

        self.nodes_with_communities={} #再次初始化

        # Community Cleaning
        m = 0
        for c in coms_to_remove:
            self.destroy_community(c)
            m += 1

        self.actual_slice += 1

    def destroy_community(self, cid):
        nodes = [x for x in self.communities[cid].keys()]
        for n in nodes:
            self.remove_from_community(n, cid)
        self.communities.pop(cid, None)

    def add_to_community(self, node, cid):

        self.g.node[node]['c_coms'][cid] = None
        if cid in self.communities:
            self.communities[cid][node] = None
        else:
            self.communities[cid] = {node: None}

    def remove_from_community(self, node, cid):
        if cid in self.g.node[node]['c_coms']:
            self.g.node[node]['c_coms'].pop(cid, None)
            if cid in self.communities and node in self.communities[cid]:
                self.communities[cid].pop(node, None)

    def centrality_test(self, subgraph):
        central = {}

        for u in subgraph.nodes():
            if u not in central:
                cflag = False
                neighbors_u = set(self.g.neighbors(u))
                if len(neighbors_u) > 1:
                    for v in neighbors_u:
                        if u > v:
                            if cflag:
                                break
                            else:
                                neighbors_v = set(self.g.neighbors(v))
                                cn = neighbors_v & neighbors_v
                                if len(cn) > 0:
                                    central[u] = None
                                    central[v] = None
                                    for n in cn:
                                        central[n] = None
                                    cflag = True
        return central
