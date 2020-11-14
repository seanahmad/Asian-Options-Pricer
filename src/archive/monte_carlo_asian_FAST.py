import numpy as np
from numba import jit
import scipy as sp
import numba as nb
S0 = 100.                 # today stock price
K = 95.                  # exercise price
T = 2.0                  # maturity in years
r = 0.15                 # risk-free rate
sigma = 0.3              # volatility (annualized)
seed = 548793    # fix a seed here
n_sims = 1000       # number of simulations
n_steps=1051200           # number of steps

@jit
def my_loop(S, Z, drift):
    for i in range(1, S.shape[0]):
        S[0,0] = S0
        for j in range(S.shape[1]):
            S[i, j] = S[i - 1, j] * np.exp(drift + Z[i-1, j])

S = np.zeros((n_steps, n_sims), dtype=np.float)
dt = T / n_steps

Z = sigma*np.sqrt(dt) * np.random.randn(n_steps - 1, n_sims)
drift = (r-0.5*sigma**2) * dt
my_loop(S, Z, drift)

# @jit(nopython=True, parallel=True)
# def asian_call_arithm(S0,K,T,r,sigma,seed,n_steps,n_sims):
#     np.random.seed(seed)
#     dt = T / n_steps
#     acall = np.zeros((n_sims), dtype=np.float64)
#     for j in range(0, n_sims):
#         St=S0
#         total = 0
#         for i in range(0,int(n_steps)):
#             e = np.random.normal()
#             St *= np.exp((r-0.5*sigma*sigma)*(dt)+sigma*e*np.sqrt(dt))
#             total += St
#             call_arithm = total/n_steps
#             acall[j] = max(call_arithm-K, 0)
#     arithm_avg = np.mean(acall) * np.exp(-r * T)
#     return arithm_avg
#
# #aavg = asian_call_arithm(S0,K,T,r,sigma,seed,n_steps,n_sims)
# #print('call price based on arithmetic average price = ', aavg)
#
# @jit(nopython=True, parallel=True)
# def asian_call_geom(S0,K,T,r,sigma,seed,n_steps,n_sims):
#     np.random.seed(seed)
#     dt = T / n_steps
#     gcall = np.zeros((n_sims), dtype=np.float64)
#     for j in range(0, n_sims):
#         St = S0
#         for i in range(0,int(n_steps)):
#             e = np.random.normal()
#             St *= np.exp((r - 0.5 * sigma * sigma) * dt + sigma * e * np.sqrt(dt))
#             Stlog = np.log(St)
#             print(St)
#             call_geom = np.exp(np.ndarray.mean(Stlog))
#             gcall[j] = max(call_geom - K, 0)
#     geom_avg = np.ndarray.mean(gcall, axis=0) * np.exp(-r * T)
#     return geom_avg

#gavg = asian_call_geom(S0,K,T,r,sigma,seed,n_steps,n_sims)
#print('call price based on arithmetic average price = ', gavg)