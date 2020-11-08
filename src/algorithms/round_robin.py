from ..salp.salp import Node

import numpy as np

class RoundRobin:
    """
   
    """
    def __init__(self, n: int, shards: np.ndarray, verbose: bool = False) -> None:
        self.shards_orig = shards
        
        self.nodes = [Node(e) for e in range(0, n)]
        
        for i, shard in enumerate(shards):
            self.nodes[i%n].shard_append(i, shard)
            if verbose:
                print("Shard", i, shard, "to Node", i%n, "node lw", self.nodes[i%n].ws)

if __name__ == '__main__':
    N = 2
    size = 2
    instance = np.array([[3, 1], [1, 3], [3, 1], [1, 3], [3, 1], [1, 3]])
    RoundRobin(N, instance, True)