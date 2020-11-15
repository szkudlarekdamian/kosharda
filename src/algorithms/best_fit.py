# from ..salp.salp import Node
from generation.gamma import Generator
from salp.salp import Node

import numpy as np


class BestFit:
    def __init__(self, n: int, shards: np.ndarray, verbose: bool, add_tolerance: bool = True, tolerance: float = 0.1):
        self.shards_orig = shards
        self.verbose = verbose
        self.node_mean_load = np.sum(shards) / n / shards.shape[1]
        if add_tolerance:
            self.node_mean_load *= (1.0 + tolerance)
        self.nodes = [Node(e) for e in range(0, n)]
        self.shard_assignments = []

    def assign(self):
        shards = sorted([{"id": i, "mean": np.mean(self.shards_orig[i]), "assigned": False} for i in
                         range(0, len(self.shards_orig))], key=lambda i: i["mean"], reverse=True)
        nodes_workloads = [{"id": i, "current_workload": 0, "assigned_shards": []} for i in range(0, len(self.nodes))]
        for node_workload in nodes_workloads:
            for shard in shards:
                if shard["assigned"]:
                    continue
                if node_workload["current_workload"] + shard["mean"] < self.node_mean_load:
                    node_workload["current_workload"] += shard["mean"]
                    shard["assigned"] = True
                    self.nodes[node_workload["id"]].shard_append(shard["id"], self.shards_orig[shard["id"]])
                    node_workload["assigned_shards"].append(shard["id"])
        if self.__unassigned_left(shards):
            unassigned_shards = self.__get_unassigned(shards)
            nodes_workloads = sorted(nodes_workloads, key=lambda i: i["current_workload"])
            for shard in unassigned_shards:
                nodes_workloads[0]["current_workload"] += shard["mean"]
                nodes_workloads[0]["assigned_shards"].append(shard["id"])
                self.nodes[nodes_workloads[0]["id"]].shard_append(shard["id"], self.shards_orig[shard["id"]])
                shard["assigned"] = True
                nodes_workloads = sorted(nodes_workloads, key=lambda i: i["current_workload"])
        if self.verbose:
            for node in nodes_workloads:
                print("Node {} - assigned shards: {}".format(node["id"], node["assigned_shards"]))

    @staticmethod
    def __unassigned_left(shards):
        for shard in shards:
            if not shard["assigned"]:
                return True
        return False

    @staticmethod
    def __get_unassigned(shards):
        unassigned = []
        for shard in shards:
            if not shard["assigned"]:
                unassigned.append(shard)
        return unassigned


if __name__ == '__main__':
    shards_number = 10
    nodes_number = 4
    gen = Generator(n=shards_number, s=12, cor=1.0, scals_range=(2.0, 4.0))
    # input_shards = gen.generate_cloud_load_vectors()
    instance = np.array([[3, 1], [3, 1], [3, 1], [1, 3], [1, 3], [1, 3]])
    best_fit = BestFit(n=nodes_number, shards=instance, verbose=True)
    best_fit.assign()
