############### Packages ###############
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

############### Constants ###############
r = 0.18
sigma = 0.3
T = 1
K = 2.0
S0 = 2.0
it = 1000000

def gbm(it, T, K, N, r, S0, sigma):
  W = np.random.normal(0, np.sqrt(T / N), (N, it))
  W = np.cumsum(W,axis=1)
  brownian = np.dot(np.tril(np.ones((N, N))), W)
  geometric_sampling = np.mean(np.exp(-r * T) * np.maximum(S0 * np.exp(sigma * np.mean(brownian, axis=0) * np.sqrt(T/N) + (r - (sigma ** 2) / 2) * (T / 2)) - K, 0))
  return geometric_sampling

gbm(1000, 1, 95, 100000, 0.15, 100, 0.3)