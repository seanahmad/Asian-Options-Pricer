############### Import Packages ###############
import numpy as np
import matplotlib.pyplot as plt

############### Constants ###############
r = 0.15                                # Expected Returns (also known as the drift coefficient
sigma = 0.3                             # Volatility (also known as the diffusion coefficient
T = 1                                   # Maturity
K = 110                                 # Strike
S0 = 150                                # Initial Stock Price
it = 10000                              # Number of Iterations
N = 1000000                             # Time Discretization
width = 28                              # Plot width
height = 12                             # Plot height

############### Geometric Average Asian Option ###############

def Geometric_Average(it, N, K, S0, r, sigma):
    D_t = np.linspace(0.,1.,N)
    W = np.cumsum(np.random.normal(0., 1., int(N))*np.sqrt(1./(N+1)))
    S_i = S0 * np.exp(sigma * W + (r - 0.5 * sigma ** 2) * D_t)
    geom_avg = []
    payoff_call = []
    payoff_put = []
    for i in range(1,it):
        average = np.exp(1/it*np.sum(np.log(S_i[:i])))
        S = S_i[:i]
        geom_avg.append(average)
        payoff_call.append(np.max(average - K, 0))
        payoff_put.append(np.max(K - average, 0))
    return geom_avg, payoff_call, payoff_put, S

average_geom = Geometric_Average(it, N, K, S0, r, sigma)[0]
payoff_call = Geometric_Average(it, N, K, S0, r, sigma)[1]
payoff_put = Geometric_Average(it, N, K, S0, r, sigma)[2]
price = Geometric_Average(it, N, K, S0, r, sigma)[3]

plt.rcParams['figure.figsize'] = (width,height)
params = {'text.color' : 'w',
          'xtick.color' : 'w',
          'ytick.color' : 'w',
          'xtick.labelsize' : 12,
          'xtick.labelsize' : 12
         }
plt.rcParams.update(params)

plt.title('Geometric Average Call with fixed μ, σ',color='w',fontsize=18)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Price, ($)',fontsize=14)

plt.plot(average_geom)
plt.show()