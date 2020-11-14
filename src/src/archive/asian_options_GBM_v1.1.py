############### Import Packages ###############
import numpy as np
import matplotlib.pyplot as plt

############### Constants ###############
r = 0.15                                # Expected Returns (also known as the drift coefficient
sigma = 0.3                             # Volatility (also known as the diffusion coefficient
T = 1                                   # Maturity
K = 2.0                                 # Strike
S0 = 100                                # Initial Stock Price
it = 1000000                            # Number of Iterations

############### Geometric Brownian Motion ###############

def GBM(it, K, T, r, S0, sigma):
    D_t = np.linspace(0.,1.,it+1)
    W = np.cumsum(np.random.normal(0., 1., int(it))*np.sqrt(1./it))
    S = []
    S.append(S0)
    for i in range(1,int(it+1)):
        S_i = S0*np.exp(sigma*np.mean(W[i-1]) + (r-0.5*sigma**2)*D_t[i])
        S.append(S_i)
    return S, D_t

Call = GBM(10000, 110, 1, 0.18, 100, 0.3)[0]
Increments = GBM(10000, 110, 1, 0.18, 100, 0.3)[1]
plt.rcParams['figure.figsize'] = (10,8)
plt.xlabel('Δt')
plt.ylabel('Price, ($)')
plt.title('Geometric Brownian Motion with fixed μ, σ')
plt.plot(Increments,Call)
plt.show()