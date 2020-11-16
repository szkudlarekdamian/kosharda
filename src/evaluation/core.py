from src.salp.salp import Node
from src.generation.base.generator import BaseGenerator

from typing import Tuple, List, Callable
import numpy as np
from tqdm import tqdm

def calculate_total_node_queueing_time(n: Node, power: float) -> float:
    stb = np.mean(n.ws)/power
    assert stb < 1, "Node {} is not stable".format(n.identity)
    coeff = stb/(1-stb)
    return coeff * np.sum(n.ws)


def calculate_all_nodes_queueing_time(nodes: List[Node], power: float) -> float:
    total_ct = 0
    for n in nodes:
        ct = calculate_total_node_queueing_time(n, power)
        total_ct += ct
    return total_ct


def evaluate_algorithms(N: int, node_power: float, algorithms: List[Tuple[str, Callable]], load_vectors: np.ndarray):
    for name, func in algorithms:
        cloud = func(N, load_vectors)
        try:
            total_ct = calculate_all_nodes_queueing_time(cloud.nodes, node_power)
        except AssertionError:
            total_ct = -1
        yield name, total_ct


def pipeline(N: int, size: int, repeats: int, cor_range: List[float], load_range: List[float], 
             generator_factory: Callable[[float], BaseGenerator], algorithms: List[Tuple[str, Callable]]):
    pbar = tqdm(total=repeats * len(cor_range) * len(load_range))
    for cor in cor_range:
        generator = generator_factory(cor)
        
        estimated_load = generator.get_estimated_cloud_load()
        estimated_node_load = estimated_load/size/N
        
        for ro in load_range:
            node_power = estimated_node_load/ro
            
            for _ in range(repeats):
                load_vectors = generator.generate_cloud_load_vectors()
                
                for res in evaluate_algorithms(N, node_power, algorithms, load_vectors):
                    yield((cor, ro) + res)
                pbar.update(1)
    pbar.close()
                
