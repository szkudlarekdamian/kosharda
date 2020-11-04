import numpy as np
from generation.gamma import Generator


class Node:
    """
    Node, which is a part of cloud.

   :param int identity: ID of a node
   :param bool active: Determines whether any more shards can be assigned to the node
   numpy.ndarray fs: List of ids of shards assigned to the node
   numpy.ndarray ws: Summed load vector of shards assigned to the node
    """

    def __init__(self, identity: int, active: bool = True) -> None:
        self.identity = identity
        self.active = active
        self.fs = np.array([], dtype=int)
        self.ws = np.array([0], dtype=float)

    def get_modules_difference(self, w: np.ndarray, nwts: np.ndarray) -> np.ndarray:
        """
        :param numpy.ndarray w: Load vector of a candidate shard
        :param numpy.ndarray nwts: Normalized load vector of a cloud
        :return numpy.ndarray: Difference between modules of [difference between node load vector and normalized cloud load vector]
                                and [also difference between the former, but node load vector includes candidate shard]
        """
        ws = self.ws.copy()
        x1 = np.linalg.norm(ws - nwts)
        x2 = np.linalg.norm(ws + w - nwts)
        return x1 - x2

    def shard_append(self, f: int, w: np.ndarray, nwts: np.ndarray) -> None:
        """
        Appends shard to the node. After this, the module of node load vector is calculated -
            when its value is greater than the module of normalized cloud load vector, the node becomes inactive.

        :param int f: ID of an appended shard
        :param numpy.ndarray w: Load vector of an appended shard
        :param numpy.ndarray nwts: Normalized load vector of a cloud
        """
        self.fs = np.append(self.fs, f)
        self.ws.resize(w.shape)
        self.ws += w
        if np.linalg.norm(self.ws) > np.linalg.norm(nwts):
            self.active = False


class SALP:
    """
    Implementation of SALP (Shards Allocation based on Load Prediction)

    :param int n: Number of nodes, to which shards will be assigned.
    :param numpy.ndarray shards: Load vectors of shards
    :param bool verbose: Verbose mode - if True, random prints appear

    wts: Summed vector of loads of the cloud
    nwts: Normalized wts (Mean load vector of a single node)
    lw: List of tuples[shard_id, shard_module], sorted in descending order by the value of shard module
    nodes: List of nodes of a cloud
    """
    def __init__(self, n: int, shards: np.ndarray, verbose: bool = False) -> None:
        self.shards_orig = shards
        self.wts = self.shards_orig.sum(axis=0)
        self.nwts = self.wts / n
        self.lw = self.__sort_shards_by_module(self.shards_orig)
        self.nodes = [Node(e) for e in range(0, n)]
        if verbose:
            self.print_basics()

        self.__arrange_shards(verbose)

    def print_basics(self) -> None:
        """Print shards, wts, nwts and lw"""
        print("----------------DESCRIPTION----------------"
              "Shards: {}\n\n\t"
              "WTS: {}\n\n\t"
              "NWTS: {}\n\n\t"
              "LW: {}\n\n\t"
              .format(self.shards_orig, self.wts, self.nwts, self.lw))

    def __sort_shards_by_module(self, shards) -> list:
        """
        :param shards: Shards to be sorted
        :return: List of tuples[shard_id, shard_module], sorted in descending order by the value of shard module
        """
        modules = [(i, np.linalg.norm(m)) for i, m in enumerate(shards)]
        return sorted(modules, key=lambda x: x[1], reverse=True)

    def __arrange_shards(self, verbose) -> None:
        """
        Assigns shards to nodes.
        Each shard is assigned to the node with the highest delta (obtained from Node.get_modules_difference).
        This function updates nodes list (self.nodes)
        :return:
        """
        if verbose:
            print("----------------ASSIGNMENT----------------")
        for shard_id, _ in self.lw:
            shard = self.shards_orig[shard_id]
            if verbose:
                print("Shard ID: {}\nShard: {}\n".format(shard_id, shard))
            max_delta = float("-inf")
            max_id = -1
            for node in self.nodes:
                if node.active:
                    delta = node.get_modules_difference(shard, self.nwts)
                    if delta > max_delta:
                        max_id = node.identity
                    if verbose:
                        print("\tNode ID: {}\n"
                              "\tDelta: {}\n"
                              .format(
                                      node.identity,
                                      delta))
            if max_id >= 0:
                self.nodes[max_id].shard_append(shard_id, shard, self.nwts)
            else:
                print("MAX_ID < 0 ({})".format(max_id))
        if verbose:
            print("----------------FINAL NODES----------------")
            for node in self.nodes:
                print("Node: {}\n"
                      "\tShards assigned [FS]: {}\n"
                      "\tLoad of node [WS]: {}\n"
                      .format(node.identity,
                              node.fs,
                              node.ws))


if __name__ == '__main__':
    _nodes = 4
    _shards_num = 10
    _load_history_length = 12
    _correlation = 1.0

    generator = Generator(_shards_num, _load_history_length, _correlation, (2.0, 4.0))
    _shards = generator.generate_cloud_load_vectors()
    
    SALP(_nodes, _shards, True)
