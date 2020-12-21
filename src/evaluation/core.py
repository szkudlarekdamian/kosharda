from src.salp.salp import Node
from src.generation.base.generator import BaseGenerator

from typing import Tuple, List, Callable
import numpy as np
from tqdm import tqdm

from numba import njit


def balance_matrix(mat: np.ndarray, power: float, thr: float = 0.95) -> np.ndarray:
    mx_lvls = np.zeros(mat.shape) + (power*thr)
    while(np.any(mat > mx_lvls)):
        dfs = mat - mx_lvls
        dfs[dfs<0] = 0
        mat = mat - dfs
        dfs = np.roll(dfs, 1, axis=1)
        mat = mat + dfs
    return mat


def get_node_vectors(N: int, load_vectors: np.ndarray, func: Callable) -> np.ndarray:
    cloud = func(N, load_vectors)
    return np.array(list(map(lambda x: x.ws, cloud.nodes)))


def evaluate_algorithm(node_power: float, nws: np.ndarray, nwts: np.ndarray) -> Tuple:
    mean_cloud_disturbance = np.mean(np.abs((nws - nwts)/nwts))

    # check if all nodes stable before balance
    means = np.mean(nws, axis=1)
    stb_vec = means/node_power

    vars = np.var(nws, axis=1)
    ca_vec = vars/(means ** 2)
    m_ca = np.mean(ca_vec)

    if np.any(stb_vec >= 0.99):  # at least one node is (almost) unstable
        # return None value for ct_vX
        return None, None, mean_cloud_disturbance, m_ca

    cff_vec = stb_vec/(1-stb_vec)
    sum_vec = np.sum(nws, axis=1)

    ct_v1 = np.sum(cff_vec * sum_vec)
    ct_v2 = np.sum(((ca_vec + 1)/2) * cff_vec * sum_vec)
        
    return ct_v1, ct_v2, mean_cloud_disturbance, m_ca


def pipeline(N: int, size: int, repeats: int, cor_range: List[float], load_range: List[float], 
             generator_factory: Callable[[float, float], BaseGenerator], algorithms: List[Tuple[str, Callable]]):
    pbar = tqdm(total=repeats * len(cor_range))
    for cor in cor_range:
        for rpt in range(repeats):
            generator = generator_factory(cor, rpt) # pass repeat number as seed

            estimated_load = generator.get_estimated_cloud_load()
            estimated_node_load = estimated_load/size/N

            load_vectors = generator.generate_cloud_load_vectors()
            nwts = load_vectors.sum(axis=0) / N
            act_load = np.sum(load_vectors)

            for name, func in algorithms:
                nws = get_node_vectors(N, load_vectors, func)

                for ro in load_range:
                    node_power = estimated_node_load/ro

                    res = evaluate_algorithm(node_power, nws, nwts)

                    yield (cor, ro, name) + res + (act_load, )
                    
            pbar.update(1) # update progress bar
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