import numpy as np
import matplotlib.pyplot as plt
import math

def random_corelated_vectors(means_vec, variances_vec, cor=1, size=100):
  assert len(means_vec) == len(variances_vec)
  cov = [[0 for _ in range(len(variances_vec))] for _ in range(len(variances_vec))] # zerowanie macierzy kowariancji
  # budowa macierzy kowariancji
  for i, cv in enumerate(variances_vec): # column
    for j, rv in enumerate(variances_vec): # row
      # na przekątnej wariancje
      # pozostałych komórkach kowariancja (korelacja zdenormalizowana)
      cov[i][j] = cov[j][i] = cv if i==j else cor*math.sqrt(cv)*math.sqrt(rv)
  # rozkład
  return np.random.multivariate_normal(means_vec, cov, size).T # python goes brrrrr