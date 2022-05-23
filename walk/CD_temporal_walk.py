import numpy as np
from scipy import stats
from random_state import random_state
from queue import PriorityQueue


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def _default_if_none(value, default, name, ensure_not_none=True):
    value = value if value is not None else default
    if ensure_not_none and value is None:
        raise ValueError(
            f"{name}: expected a value to be specified in either `__init__` or `run`, found None in both"
        )
    return value

def naive_weighted_choices(rs, weights, size=None):
    probs = np.cumsum(weights)
    total = probs[-1]
    if total == 0:
        # all weights were zero (probably), so we shouldn't choose anything
        return None

    thresholds = rs.random() if size is None else rs.random(size)
    idx = np.searchsorted(probs, thresholds * total, side="left")

    return idx

class GraphWalk(object):
    def __init__(self, graph, graph_schema=None, seed=None):
        self.graph = graph
        # Initialize the random state
        self._check_seed(seed)
        self._random_state, self._np_random_state = random_state(seed)

    def _check_seed(self, seed):
        if seed is not None:
            if type(seed) != int:
                self._raise_error(
                    "The random number generator seed value, seed, should be integer type or None."
                )
            if seed < 0:
                self._raise_error(
                    "The random number generator seed value, seed, should be non-negative integer or None."
                )

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.
        """
        if seed is None:
            # Use the class's random state
            self._random_state, self._np_random_state = random_state(seed)
            return self._random_state, self._np_random_state
        # seed the random number generators
        return random_state(seed)

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def _raise_error(self, msg):
        raise ValueError("({}) {}".format(type(self).__name__, msg))

class CD_temporal_walk(GraphWalk):
    def __init__(
        self,
        graph,
        data,
        cw_size=None,
        max_walk_length=80,
        initial_edge_bias=None,
        walk_bias=None,
        p_walk_success_threshold=0.01,
        seed=None,

        nodes_com_d=None,
        com_nodes_d=None,
        communities_wight=2.0
    ):
        self.cw_size = cw_size
        self.max_walk_length = max_walk_length
        self.initial_edge_bias = initial_edge_bias
        self.walk_bias = walk_bias
        self.p_walk_success_threshold = p_walk_success_threshold

        self.graph=graph
        self.data=data
        self.nodes_com_d=nodes_com_d #node：com
        self.com_nodes_d=com_nodes_d #com：{node}
        self.communities_wight=communities_wight
    def run(
        self,
        num_cw,
        cw_size=None,
        max_walk_length=None,
        initial_edge_bias=None,
        walk_bias=None,
        p_walk_success_threshold=None,
        seed=None,
    ):
        cw_size = _default_if_none(cw_size, self.cw_size, "cw_size")
        max_walk_length = _default_if_none(
            max_walk_length, self.max_walk_length, "max_walk_length"
        )
        initial_edge_bias = _default_if_none(
            initial_edge_bias,
            self.initial_edge_bias,
            "initial_edge_bias",
            ensure_not_none=False,
        )
        walk_bias = _default_if_none(
            walk_bias, self.walk_bias, "walk_bias", ensure_not_none=False
        )
        p_walk_success_threshold = _default_if_none(
            p_walk_success_threshold,
            self.p_walk_success_threshold,
            "p_walk_success_threshold",
        )

        if cw_size < 2:
            raise ValueError(
                f"cw_size: context window size should be greater than 1, found {cw_size}"
            )
        if max_walk_length < cw_size:
            raise ValueError(
                f"max_walk_length: maximum walk length should not be less than the context window size, found {max_walk_length}"
            )
        _, np_rs = self._get_random_state(seed)

        walks = []
        num_cw_curr = 0
        self.data.index = range(0,len(self.data)) #保证索引由0开始，方便_sample对应
        sources = self.data['source']
        targets = self.data['target']
        times = self.data['time']
        edge_biases = self._temporal_biases(times, None, bias_type=initial_edge_bias, is_forward=False,)
        successes = 0
        failures = 0

        def not_progressing_enough():
            posterior = stats.beta.ppf(0.95, 1 + successes, 1 + failures)
            return posterior < p_walk_success_threshold

        while num_cw_curr < num_cw:
            first_edge_index = self._sample(len(times), edge_biases, np_rs)
            
            src = sources[first_edge_index]
            dst = targets[first_edge_index]
            t = times[first_edge_index]
            
            remaining_length = num_cw - num_cw_curr + cw_size - 1

            walk = self._CD_walk(src, dst, t, min(max_walk_length, remaining_length), walk_bias, np_rs)
            if len(walk) >= cw_size:
                walks.append(walk)
                num_cw_curr += len(walk) - cw_size + 1
                successes += 1
            else:
                failures += 1
                if not_progressing_enough():
                    raise RuntimeError(
                        f"Discarded {failures} walks out of {failures + successes}. "
                        f"Consider using a smaller context window size (currently cw_size={cw_size})."
                    )
        print('successes',successes)
        print('failures',failures)
        return walks

    def _sample(self, n, biases, np_rs):
        if biases is not None:
            assert len(biases) == n
            return naive_weighted_choices(np_rs, biases)
        else:
            return np_rs.choice(n)

    def _exp_biases(self, times, t_0, decay):
        # t_0 assumed to be smaller than all time values
        return softmax(t_0 - np.array(times) if decay else np.array(times) - t_0)

    def _temporal_biases(self, times, time, bias_type, is_forward):
        if bias_type is None:
            # default to uniform random sampling
            return None

        # time is None indicates we should obtain the minimum available time for t_0
        t_0 = time if time is not None else min(times)

        if bias_type == "exponential":
            # exponential decay bias needs to be reversed if looking backwards in time
            return self._exp_biases(times, t_0, decay=is_forward)
        else:
            raise ValueError("Unsupported bias type")

    def _CD_walk(self, src, dst, t, length, bias_type, np_rs):
        list_communites = self.nodes_com_d[src] 
        same_communites_nodesList = {}
        for com_id in list_communites:
            communities_nodesList = self.com_nodes_d[com_id]
            for nodes in communities_nodesList:
                if(same_communites_nodesList.get(nodes) == None):
                    same_communites_nodesList[nodes]=1

        walk = [src, dst]
        node, time = dst, t
        for _ in range(length - 2):
            result = self._next_step(node, time=time, bias_type=bias_type, np_rs=np_rs,same_communites_nodesList=same_communites_nodesList)
            if result is not None:
                node, time = result
                walk.append(node)
            else:
                break #找不到next了
        return walk

    def _next_step(self, node, time, bias_type, np_rs, same_communites_nodesList):
        neighbours, times = self._neighbor_arrays(node)
        index=0
        for id,t in enumerate(times):
            if(t>=time):
                index = id
                break;
            
        neighbours = neighbours[index:]
        times = times[index:]
        if len(neighbours) > 0:
            biases = self._temporal_biases(times, time, bias_type, is_forward=True)
            for node_index,node_id in enumerate(neighbours):
                if(same_communites_nodesList.get(node_id) == 1):
                    biases[node_index]=biases[node_index]*self.communities_wight
            
            chosen_neighbour_index = self._sample(len(neighbours), biases, np_rs)
            assert chosen_neighbour_index is not None

            next_node = neighbours[chosen_neighbour_index]
            next_time = times[chosen_neighbour_index]
            return next_node, next_time
        else:
            return None

    def _neighbor_arrays(self,node):
        qr = PriorityQueue()
        neighbor_nodes= self.graph.adj.get(node)
        time = []
        neighbors = []
        for nbr, eattr in neighbor_nodes.items():
            for id in eattr:
                qr.put((eattr[id]['time'],nbr))

        time = []
        neighbors = []
        while not qr.empty():
            item = qr.get()
            time.append(item[0])
            neighbors.append(item[1])
        
        return neighbors,time