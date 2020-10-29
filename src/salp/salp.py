import numpy as np
from typing import Tuple
from generation.gamma import Generator


class Endpoint:
    def __init__(self, identity: int):
        self.identity = identity
        self.active = True
        # shards
        self.fs = np.array([])
        # load vector
        self.ws = np.array([0.])

    def get_modules_difference(self, w, nwts):
        ws = self.ws.copy()
        x1 = np.linalg.norm(ws - nwts)
        x2 = np.linalg.norm(ws + w - nwts)
        return x1 - x2

    def shard_append(self, f, w, nwts):
        self.fs = np.append(self.fs, f)
        self.ws.resize(w.shape)
        self.ws += w
        if np.linalg.norm(self.ws) > np.linalg.norm(nwts):
            self.active = False


class SALP:
    def __init__(self, n: int, f: int, s: int, cor: float, scals_range: Tuple[float, float]) -> None:
        generator = Generator(f, s, cor, scals_range)
        self.shards_orig = generator.generate_cloud_load_vectors()
        self.wts = self.shards_orig.sum(axis=0)
        self.nwts = self.wts / n
        self.lw = self.sort_shards_by_module(self.shards_orig)
        self.endpoints = [Endpoint(e) for e in range(0, n)]

        self.print_basics()

        self.arrange_shards()

    def print_basics(self):
        print("\tShards: {}\n\n\t"
              "WTS: {}\n\n\t"
              "NWTS: {}\n\n\t"
              "LW: {}\n\n\t"
              .format(self.shards_orig, self.wts, self.nwts, self.lw))


    def sort_shards_by_module(self, shards):
        modules = [(i, np.linalg.norm(m)) for i, m in enumerate(shards)]
        return sorted(modules, key=lambda x: x[1], reverse=True)

    def arrange_shards(self):
        for shard_id, _ in self.lw:
            shard = self.shards_orig[shard_id]
            print(shard_id)
            max_delta = float("-inf")
            max_id = -1
            for endpoint in self.endpoints:
                if endpoint.active:
                    delta = endpoint.get_modules_difference(shard, self.nwts)
                    if delta > max_delta:
                        max_id = endpoint.identity
                    print("Shard: {}\n"
                          "Endpoint: {}\n"
                          "Delta: {}\n\n"
                          .format(shard,
                                  endpoint.identity,
                                  delta))
                else:
                    print("ENDPOINT {} NOT ACTIVE".format(endpoint.identity))
            if max_id >= 0:
                self.endpoints[max_id].shard_append(shard_id, shard, self.nwts)
            else:
                print("MAX_ID < 0 ({})".format(max_id))
        for endpoint in self.endpoints:
            print("Endpoint: {}\n"
                  "Shards: {}\n"
                  "Load: {}\n\n"
                  .format(endpoint.identity,
                          endpoint.fs,
                          endpoint.ws))


if __name__ == '__main__':
    endpoints = 4
    shards = 10
    shards_history = 12
    correlation = 1.0

    SALP(endpoints, shards, shards_history, correlation, (2.0, 4.0))
