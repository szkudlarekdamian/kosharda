from src.salp.salp import Node
from src.generation.base.generator import BaseGenerator

from typing import Tuple, List, Callable
import numpy as np
from tqdm import tqdm


def balance_matrix(mat: np.ndarray, power: float, thr: float = 0.95) -> np.ndarray:
    mx_lvls = np.zeros(mat.shape) + (power*thr)
    while(np.any(mat > mx_lvls)):
        dfs = mat - mx_lvls
        dfs[dfs<0] = 0
        mat = mat - dfs
        dfs = np.roll(dfs, 1, axis=1)
        mat = mat + dfs
    return mat


def evaluate_algorithms(N: int, node_power: float, algorithms: List[Tuple[str, Callable]], load_vectors: np.ndarray, nwts: np.ndarray):
    for name, func in algorithms:
        cloud = func(N, load_vectors)
        nws = np.array(list(map(lambda x: x.ws, cloud.nodes)))

        mean_cloud_disturbance = np.mean(np.abs((nws - nwts)/nwts))

         # check if all nodes stable before balance
        means = np.mean(nws, axis=1)
        stb_vec = means/node_power

        if np.any(stb_vec >= 0.99):  # at least one node is (almost) unstable
            # print('oh shit', name, np.max(means/node_power))
            yield name, None, mean_cloud_disturbance # return None value for total_ct
            continue

        cff_vec = stb_vec/(1-stb_vec)
        vars = np.var(nws, axis=1)
        tq_vec = vars/(means ** 2)

        total_ct = np.sum(tq_vec * cff_vec)
        
        yield name, total_ct, mean_cloud_disturbance


def pipeline(N: int, size: int, repeats: int, cor_range: List[float], load_range: List[float], 
             generator_factory: Callable[[float, float], BaseGenerator], algorithms: List[Tuple[str, Callable]]):
    pbar = tqdm(total=repeats * len(cor_range) * len(load_range))
    # sed = 0 # random seed for generators
    for cor in cor_range:
        for ro in load_range:
            for rpt in range(repeats):
                # print(cor, ro, sed)
                generator = generator_factory(cor, rpt) # pass repeat number as seed
        
                estimated_load = generator.get_estimated_cloud_load()
                estimated_node_load = estimated_load/size/N
                node_power = estimated_node_load/ro

                load_vectors = generator.generate_cloud_load_vectors()
                nwts = load_vectors.sum(axis=0) / N
                act_load = np.sum(load_vectors)

                for res in evaluate_algorithms(N, node_power, algorithms, load_vectors, nwts):
                    yield (cor, ro) + res + (act_load, )

                pbar.update(1) # update progress bar
                # sed += 1 # increment the random seed
    pbar.close()


if __name__ == '__main__':
    x1 = np.array([
        [0,1,2], 
        [3,4,5]
        ])
    x2 = np.array([
        [2,0,1], 
        [5,3,4]
        ])
    assert np.all(x2 == np.roll(x1, 1, axis=1))

    p1 = np.array([
        [9,0],
        [0,9]
    ])
    pwr = 5
    thr = 0.95
    p2 = balance_matrix(p1, pwr, thr)

    expected = np.array([
        [thr*pwr, 9 - thr*pwr],
        [9 - pwr*thr, thr*pwr]
    ])
    assert np.all(p2 == expected)

    p3 = np.array([
        [9, 4, 0],
        [0, 9, 4]
    ])

    p4 = balance_matrix(p3, pwr, thr)
    assert np.all(p4 <= thr*pwr)
    assert np.sum(p3) == np.sum(p4)