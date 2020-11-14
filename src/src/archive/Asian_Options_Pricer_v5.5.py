# -*- coding: utf-8 -*-
#region Summary
"""
# Asian Options Pricing & Hedging Tool

Course: FE 620: Pricing & Hedging | Stevens Institute of Technology\
Advisor: Dan Pirjol\
Group: Theo Dimitrasopoulos, Will Kraemer, Vaikunth Seshadri, Snehal Rajguru\

Link: https://colab.research.google.com/drive/1g9xDGWCoKgFhNQWMW_nGfeJ8C1_5zaOE\
*Version: v5.5*
"""
#endregion

#region Import Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
#endregion

#region Constants
r = 0.15                                # Expected Returns (also known as the drift coefficient
q = 0.0                                 # Dividend Yield Rate
T = 1                                   # Maturity
K = 90                                  # Strike
S0 = 100                                # Initial Underlying Price
sigma = 0.3                             # Volatility (also known as the diffusion coefficient
it = 1000                               # Number of Iterations for Monte Carlo Simulation
N = 10000                               # Time Discretization
width = 18                              # Plot width
height = 12                             # Plot height
#endregion

#region Plot Parameters
plt.rcParams['figure.figsize'] = (width,height)
params = {'text.color' : 'black',
          'xtick.color' : 'black',
          'ytick.color' : 'black',
          'xtick.labelsize' : 12,
          'ytick.labelsize' : 12,
          'legend.loc' : 'upper right',
         }
plt.rcParams.update(params)
#endregion

#region Geometric Exact Pricing Definition
def geom_exact(r,q,T,K,S0,sigma):
  G0 = S0 * np.exp(0.5 * (r - q - (sigma**2)/6) * T - (1/12) * (sigma**2) * T)
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * T)
  d2 = (1/(Sigma_G * np.sqrt(T))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * T)
  G_c = np.exp(-r * T) * (G0 * norm.cdf(d1) - K * norm.cdf(d2))
  G_p = np.exp(-r * T) * (K * norm.cdf(-d2) - G0 * norm.cdf(-d1))
  return G_c, G_p

print('The exact call Asian option price with geometric averaging is: ',geom_exact(r,q,T,K,S0,sigma)[0])
print('The exact put Asian option price with geometric averaging is: ',geom_exact(r,q,T,K,S0,sigma)[1])
#endregion

#region Geometric Brownian Motion Definition
def monte_carlo_gbm(it, r, S0, sigma):
  D_t = np.linspace(0.,1.,N)
  W = np.cumsum(np.random.normal(0., np.sqrt(D_t), int(N))*np.sqrt(1./(N)))
  S_i = S0 * np.exp(sigma * W + (r - q - 0.5 * sigma ** 2) * D_t)
  S = []
  S.append(S0)
  for i in range(1,N+1):
    S = S_i[:i]
  return S, D_t
#endregion

#region Geometric Average Asian Option: Monte Carlo Definition
def monte_carlo_geometric(it, N, K, S0, r, sigma):
  D_t = np.linspace(0.,1.,N)
  W = np.cumsum(np.random.normal(0., np.sqrt(D_t), int(N))*np.sqrt(1./(N)))
  S_i = S0 * np.exp(sigma * W + (r - q - 0.5 * sigma ** 2) * D_t)
  geom_avg = []
  payoff_call = []
  payoff_put = []
  for i in range(1,it+1):
    S = S_i[:i]
    geom_avg.append(np.exp(1/it*np.sum(np.log(S_i[:i]))))
    payoff_call.append(np.max((np.exp(1/it*np.sum(np.log(S_i[:i])))) - K, 0))
    payoff_put.append(np.max(K - (np.exp(1/it*np.sum(np.log(S_i[:i])))), 0))
  return geom_avg, payoff_call, payoff_put, S
#endregion

#region Strike Vector for Plotting
strike = np.linspace(K-30,K+30,it)
#endregion

#region Plot Data
average_geom = monte_carlo_geometric(it, N, K, S0, r, sigma)[0]
payoff_call = monte_carlo_geometric(it, N, K, S0, r, sigma)[1]
payoff_put = monte_carlo_geometric(it, N, K, S0, r, sigma)[2]
price = monte_carlo_geometric(it, N, K, S0, r, sigma)[3]
#endregion

#region Geometric Average Price

'''Geometric Average Price vs. Spot Price'''
avg_vs_spot = plt.figure()
plt.title('Geometric Average vs. Spot Price',color='black',fontsize=18)
plt.xlabel('Spot Price ($)',color='black',fontsize=14)
plt.ylabel('Geometric Average ($)',color='black',fontsize=14)
plt.plot(average_geom,price)
plt.show()

'''Geometric Average Price'''
g_avg = plt.figure()
plt.title('Geometric Average',color='black',fontsize=18)
plt.xlabel('Iterations',color='black',fontsize=14)
plt.ylabel('Geometric Average ($)',color='black',fontsize=14)
plt.plot(average_geom)
plt.show()
#endregion

#region Geometric Average Call Option

'''Call payoff vs. Spot Price'''
callpayoff_vs_spot = plt.figure()
plt.title('Call Option Payoff vs. Spot Price',color='black',fontsize=18)
plt.xlabel('Spot Price ($)',color='black',fontsize=14)
plt.ylabel('Payoff ($)',color='black',fontsize=14)
plt.plot(payoff_call,price)
plt.show()

'''Call payoff'''
callpayoff = plt.figure()
plt.title('Call Option Payoff',color='black',fontsize=18)
plt.xlabel('Iterations',color='black',fontsize=14)
plt.ylabel('Payoff ($)',color='black',fontsize=14)
plt.plot(payoff_call)
plt.show()

'''Call vs. Strike'''
callpayoff_vs_strike = plt.figure()
plt.title('Call Option Payoff vs. Strike',color='black',fontsize=18)
plt.xlabel('Strike ($)',color='black',fontsize=14)
plt.ylabel('Payoff ($)',color='black',fontsize=14)
plt.plot(payoff_call,np.abs(strike))
plt.show()
#endregion

#region Geometric Average Put Option

'''Put payoff vs. Spot Price'''
putpayoff__vs_spot = plt.figure()
plt.title('Put Option Payoff vs. Spot Price',color='black',fontsize=18)
plt.xlabel('Spot Price ($)',color='black',fontsize=14)
plt.ylabel('Payoff ($)',color='black',fontsize=14)
plt.plot(payoff_put,price)
plt.show()

'''Put payoff'''
putpayoff = plt.figure()
plt.title('Put Option Payoff',color='black',fontsize=18)
plt.xlabel('Iterations',color='black',fontsize=14)
plt.ylabel('Payoff ($)',color='black',fontsize=14)
plt.plot(payoff_put)
plt.show()

'''Put vs. Strike'''
putpayoff_vs_strike = plt.figure()
plt.title('Put Option Payoff vs. Strike',color='black',fontsize=18)
plt.xlabel('Strike ($)',color='black',fontsize=14)
plt.ylabel('Payoff ($)',color='black',fontsize=14)
plt.plot(payoff_put,strike)
plt.show()
#endregion

#region Pricer Benchmarking

'''Generate Random Si'''
Dt = monte_carlo_gbm(N, r, S0, sigma)[1]
S1 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S2 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S3 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S4 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S5 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S6 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S7 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S8 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S9 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]
S10 = monte_carlo_gbm(N, np.random.uniform(0.1,0.6), np.random.uniform(90.0,500.0), np.random.uniform(0.01,0.4))[0]

'''Random Si plots'''
market_data_random = plt.figure()
plt.plot(Dt,S1,label='Stock 1')
plt.plot(Dt,S2,label='Stock 2')
plt.plot(Dt,S3,label='Stock 3')
plt.plot(Dt,S4,label='Stock 4')
plt.plot(Dt,S5,label='Stock 5')
plt.plot(Dt,S6,label='Stock 6')
plt.plot(Dt,S7,label='Stock 7')
plt.plot(Dt,S8,label='Stock 8')
plt.plot(Dt,S9,label='Stock 9')
plt.plot(Dt,S10,label='Stock 10')

plt.title('Randomly generated paths for 10 stocks', color='black', fontsize=18)
plt.xlabel('Time (Dt)',color='black',fontsize=14)
plt.ylabel('Spot Price ($)',color='black',fontsize=14)
plt.legend(("Stock 1",
            'Stock 2',
            'Stock 3',
            'Stock 4',
            'Stock 5',
            'Stock 6',
            'Stock 7',
            'Stock 8',
            'Stock 9',
            'Stock 10'
            ),
           fontsize=10
           )
plt.show()
#endregion

#region Data Collection
market_data_real = plt.figure()
msft_adj_close = pdr.get_data_yahoo("MSFT",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(msft_adj_close,label='Microsoft (MSFT)')

aapl_adj_close = pdr.get_data_yahoo("AAPL",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(aapl_adj_close,label='Apple (AAPL)')

gs_adj_close = pdr.get_data_yahoo("GS",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(gs_adj_close,label='Goldman Sachs (GS)')

nflx_adj_close = pdr.get_data_yahoo("NFLX",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(nflx_adj_close,label='Netflix (NFLX)')

fb_adj_close = pdr.get_data_yahoo("FB",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(fb_adj_close,label='Facebook (FB)')

twtr_adj_close = pdr.get_data_yahoo("TWTR",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(twtr_adj_close,label='Twitter (TWTR)')

ibm_adj_close = pdr.get_data_yahoo("IBM",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(ibm_adj_close,label='IBM (IBM)')

baba_adj_close = pdr.get_data_yahoo("BABA",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(baba_adj_close,label='Alibaba Group (BABA)')

ba_adj_close = pdr.get_data_yahoo("BA",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(ba_adj_close,label='Boeing (BA)')

cme_adj_close = pdr.get_data_yahoo("CME",start='2013-1-1',end='2020-12-31')['Adj Close']
plt.plot(cme_adj_close,label='Amazon (CME)')

plt.title('Adjusted Close for Selected Stocks',color='black',fontsize=18)
plt.xlabel('Date',color='black',fontsize=14)
plt.ylabel('Stock Price ($)',color='black',fontsize=14)
plt.legend()
plt.show()
#endregion