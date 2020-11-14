############### Packages ###############
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance

############### Constants ###############
r = 0.18
sigma = 0.3
T = 1
K = 2.0
S0 = 2.0
iterations = 1000000

############### Pricing Formulas ###############
def arithmeticExactPricing(r,T,K,S0,sigma):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)

def geometricExactPricing(r,T,K,S0,sigma):
    d1 = (np.log(S0/K) + (r+(sigma**2)/6)*T/2)/(sigma*np.sqrt(T/3))
    d2 = (np.log(S0/K) + (r-(sigma**2)/2)*T/2)/(sigma*np.sqrt(T/3))
    return S0*np.exp(-(r + (sigma**2)/6)*T/2) * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)

############### Calculate the expected value of the lognormal distribution ###############
def expectedval(r, sigma, T, S0):
    return ((np.exp(r*T)-1)*S0)/(r*T)

############### Calculate the variance of the lognormal distribution ###############
def variance(r, sigma, T, S0):
    temp1 = (2*S0**2/((T**2)*(r+sigma**2)))
    temp2 = (np.exp((2*r+sigma**2)*T)-1)/(2*r+sigma**2)
    temp3 = (np.exp(r*T)-1)/r
    temp = temp1 * (temp2 - temp3)
    return temp - (((np.exp(r*T)-1)*S0)/(r*T))**2


def ecarttype(r, sigma, T, S0):
    return np.sqrt(variance(r, sigma, T, S0))


esp = expectedval(r, sigma, T, S0)
var = variance(r, sigma, T, S0)
EC = ecarttype(r, sigma, T, S0)

mulog = np.log(esp**2/np.sqrt(var + esp**2))
sigmalog = np.sqrt(np.log(1 + (var/(esp**2))))

############### Monte Carlo Simulation ###############

def Monte_Carlo_lognormal(iterations, K, T, r):
    lognorm = np.random.lognormal(mulog, sigmalog, iterations)
    Price = np.exp(-r*T) * np.maximum(lognorm - K, 0)
    std = np.std(Price)
    mean = np.mean(Price)
    return mean+(1.96*std/np.sqrt(iterations)), mean, mean-(1.96*std/np.sqrt(iterations))


def Monte_Carlo_lognormal_it(iterations, K, T, r):
    lognorm = np.random.lognormal(mulog, sigmalog, iterations)
    Price = np.exp(-r*T) * np.maximum(lognorm - K, 0)
    lower_list = []
    upper_list = []
    mean_list = []
    for i in range(1, iterations, 100):
        std = np.std(Price[:i])
        mean = np.mean(Price[:i])
        mean_list.append(mean)
        upper_list.append(mean+(1.96*std/np.sqrt(i)))
        lower_list.append(mean-(1.96*std/np.sqrt(i)))
    return upper_list, mean_list, lower_list

#PIRJOL COMMENTS:
#W(i+1) = W(i) + \sqrt{\Delta T} \epsilon where \Delta T = T/n is the time step, and \epsilon = N(0,1) are iid random normally distributed variables.
#Then we construct the geometric Brownian motion as S(i+1) = S(i) * exp(sigma*\sqrt{\Delta t}*\epsilon + ( r - 1/2 \sigma^2) * \Delta t )

############### Call Monte Carlo Simulation ###############

data_lognormal = Monte_Carlo_lognormal_it(iterations, K, T, r)

############### Clean Monte Carlo Simulation Data ###############

data = list(data_lognormal)
data_upper = data[0]
data_mean = data[1]
data_lower = data[2]

############### Plot Monte Carlo Simulation Data ###############

# Prune the initial 13 values to get a clearer view of the main line
plt.plot(data_upper[12:], label = 'Upper Confidence Interval')
# Prune the initial 6 values to get a clearer view of the main line
plt.plot(data_mean[5:], label = 'Mean')
# Prune the initial 29 values to get a clearer view of the main line
plt.plot(data_lower[28:], label = 'Lower Confidence Interval')
plt.ylabel('Expected Value')
plt.xlabel('Iterations')
plt.legend()
plt.show()

############### Exact Arithmetic & Geometric Pricing ###############
print('The price determined by the Monte Carlo Simulation converges at approximately: 0.219')
print('The exact price using the arithmetic mean is:', arithmeticExactPricing(r,T, K,S0, sigma))
print('The exact price using the geometric mean is:', geometricExactPricing(r,T, K,S0, sigma))

#SCRATCHPAD INPUTS

#PIRJOL COMMENTS:
#W(i+1) = W(i) + \sqrt{\Delta T} \epsilon where \Delta T = T/n is the time step, and \epsilon = N(0,1) are iid random normally distributed variables.
#Then we construct the geometric Brownian motion as S(i+1) = S(i) * exp(sigma*\sqrt{\Delta t}*\epsilon + ( r - 1/2 \sigma^2) * \Delta t )

def Brownian_it(iterations, T, sigma):
    w_0 = 0.0
    S_0=0.0
    mean_list = []
    brownian = []
    for i in range(1, iterations, 100):
        epsilon = np.random.normal(0,1)
        delta_t = T/iterations
        w_1 = w_0 + np.sqrt(delta_t)*epsilon
        mean_list.append(w_1)
        S_1 = S_0*np.exp(sigma*np.sqrt(delta_t))*epsilon + (r-1/2*sigma**2*delta_t)
        brownian.append(S_1)
        S_0 = S_1
    return brownian

#SCRATCHPAD OUTPUTS
plt.plot(Brownian_it(iterations,T,sigma))
print(Brownian_it(iterations,T,sigma))