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
import matplotlib.pyplot as plt
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
width = 16                              # Plot width
height = 12                             # Plot height
#endregion

#region Plot Parameters
plt.rcParams['figure.figsize'] = (width,height)
params = {'text.color' : 'black',
          'xtick.color' : 'black',
          'ytick.color' : 'black',
          'xtick.labelsize' : 12,
          'ytick.labelsize' : 12,
          'legend.loc' : 'upper left',
         }
plt.rcParams.update(params)
#endregion

#region Data Collection
market_data_real = plt.figure()
msft_adj_close = pdr.get_data_yahoo("MSFT",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(msft_adj_close,label='Microsoft (MSFT)')

aapl_adj_close = pdr.get_data_yahoo("AAPL",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(aapl_adj_close,label='Apple (AAPL)')

gs_adj_close = pdr.get_data_yahoo("GS",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(gs_adj_close,label='Goldman Sachs (GS)')

nflx_adj_close = pdr.get_data_yahoo("NFLX",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(nflx_adj_close,label='Netflix (NFLX)')

fb_adj_close = pdr.get_data_yahoo("FB",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(fb_adj_close,label='Facebook (FB)')

twtr_adj_close = pdr.get_data_yahoo("TWTR",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(twtr_adj_close,label='Twitter (TWTR)')

tsla_adj_close = pdr.get_data_yahoo("TSLA",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(tsla_adj_close,label='Tesla (TSLA)')

baba_adj_close = pdr.get_data_yahoo("BABA",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(baba_adj_close,label='Alibaba Group (BABA)')

c_adj_close = pdr.get_data_yahoo("C",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(c_adj_close,label='Citigroup (C)')

cme_adj_close = pdr.get_data_yahoo("CME",start='1927-1-1',end='2020-12-31')['Adj Close']
plt.plot(cme_adj_close,label='Amazon (CME)')

plt.title('Adjusted Close for Selected Stocks',color='black',fontsize=18)
plt.xlabel('Date',color='black',fontsize=14)
plt.ylabel('Stock Price ($)',color='black',fontsize=14)
plt.legend()
plt.show()
#endregion