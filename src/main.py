import src.generation.gamma as ga
import src.generation.multivariate_normal as mn

from src.salp.salp import SALP

from src.algorithms.round_robin import RoundRobin
from src.algorithms.best_fit import BestFit

from src.evaluation.core import pipeline

import numpy as np
import pandas as pd


if __name__ == '__main__':
    # SALP
    def salp_wrapper(N: int, load_vectors):
        return SALP(N, load_vectors, False)
    # Best Fit
    def bf_wrapper(N: int, load_vectors):
        bf = BestFit(N, load_vectors, False, False)
        bf.assign()
        return bf
    # Round robin
    def rr_wrapper(N: int, load_vectors):
        return RoundRobin(N, load_vectors, False)

    algorithms = [
        ('SALP', salp_wrapper),
        ('BF', bf_wrapper),
        ('RR', rr_wrapper),
    ]

    N = 100
    F = 10*N
    cor_range = np.arange(0.0, 1.01, 0.05)
    load_range = np.arange(0.7, 0.91, 0.05)
    size=100
    repeats=100
    rng = (2,4)

    def gamma_generator_factory(cor):
        return ga.Generator(F, size, cor, rng)

    gen = pipeline(N, size, repeats, cor_range, load_range, gamma_generator_factory, algorithms)
    df = pd.DataFrame(gen, columns=['correlation', 'load', 'algorithm', 'value'])

    df.to_csv('results/N{}-F{}-S{}-R{}-result-v1.csv'.format(N,F,size,repeats), index=False)
    
