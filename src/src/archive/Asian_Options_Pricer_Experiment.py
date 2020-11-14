############### Packages ###############
import numpy as np
import matplotlib.pyplot as plt

############### Constants ###############
r = 0.15
sigma = 0.3
T = 1.0
K = 95
S0 = 100
it = 1000
N = 10000


def monte_carlo_call_matrix(it, T, K, N, r, S0, sigma):
  geometric_sampling = []
  geometric_sampling.append(np.exp(-r * T) * np.maximum(S0 - K, 0))
  W = np.cumsum(np.random.normal(0, 1, (it, N)),axis=1)
  geometric_sampling.append(np.exp(-r * T) * np.maximum(S0 * np.exp(sigma * np.mean(W, axis=0) * np.sqrt(T/N) + (r - (sigma ** 2) / 2) * (T / N)) - K, 0))
  return geometric_sampling

call_prices = monte_carlo_call_matrix(it, T, K, N, r, S0, sigma)
