from ..generation.gamma import Generator
from ..salp.salp import Node

import numpy as np


class BestFit:
    def __init__(self, n: int, shards: np.ndarray, verbose):
        self.shards_orig = shards
        self.verbose = verbose
        self.node_mean_load = np.sum(shards) / n / len(shards)
        self.nodes = [Node(e) for e in range(0, n)]
        self.shard_assignments = []

    def assign_shards(self):
        current_node_load = np.full(shape=len(self.nodes), fill_value=self.node_mean_load, dtype=float)
        shards_mean_loads = [np.mean(shard) for shard in self.shards_orig]
        if self.verbose:
            print("Node mean load", self.node_mean_load)
            print("Shard mean loads", shards_mean_loads)
            print("Shard mean loads sum - {}".format(np.sum(shards_mean_loads)))
        for shard_id, mean_load in enumerate(shards_mean_loads):
            tmp_node_load = [[i, current_node_load[i] - mean_load] for i in range(len(self.nodes))]
            node = sorted(self._filter_load_vector(tmp_node_load), key=lambda x: x[1])[0]
            current_node_load[node[0]] -= mean_load
            self.nodes[node[0]].shard_append(shard_id, self.shards_orig[shard_id])
            self.shard_assignments.append(node[0])
            if self.verbose:
                print("\nCurrent shard's id - {}".format(shard_id))
                print("Current shard's mean - {}".format(mean_load))
                print("Best fit node after assignment\n", node)
                print("Current nodes loads after assignment\n", current_node_load)
        if self.verbose:
            self.describe_assignment()

    @staticmethod
    def _filter_load_vector(load_vector):
        filtered_vector = []
        for load in load_vector:
            if load[1] > 0:
                filtered_vector.append(load)
        return filtered_vector

    def describe_assignment(self):
        print()
        for i in range(len(self.nodes)):
            s = []
            for pos, a in enumerate(self.shard_assignments):
                if a == i:
                    s.append(pos)
            print("Shards assigned to node {} - {}".format(i, s))


if __name__ == '__main__':
    shards_number = 10
    nodes_number = 4
    gen = Generator(n=shards_number, s=12, cor=1.0, scals_range=(2.0, 4.0))
    input_shards = gen.generate_cloud_load_vectors()
    best_fit = BestFit(n=nodes_number, shards=input_shards, verbose=True)
