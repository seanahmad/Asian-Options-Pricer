# -*- coding: utf-8 -*-

# Install packages
'''
!pip install -q numpy
!pip install -q matplotlib
!pip install -q scipy
!pip install -q quandl
!pip install -q yfinance
!pip install -q pandas
!pip install quantsbin
'''
# Import Packages
import math
import random
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Initial Underlying Price
S0 = 100#@param {type:"number"}

# Risk-free rate (also known as the drift coefficient)
r = 0.15#@param {type:"number"}

# Dividend Yield Rate
q = 0.0#@param {type:"number"}

# Valuation Date
t = 0.0#@param {type:"number"}

# Maturity
T = 1.0#@param {type:"number"}

# Strike
K = 95#@param {type:"number"}

# Volatility (also known as the diffusion coefficient)
sigma = 0.3#@param {type:"number"}

# Number of Iterations for Monte Carlo Simulation
iterations = 100000#@param {type:"integer"}

# Time Steps
timesteps = 100000#@param {type:"integer"}

# Random Seed
seed_user = 1764#@param {type:"integer"}

# Universal pseudorandom number
seed = np.random.seed(seed_user)

"""### Plot Variables"""

# Universal Plot width
width = 25 #@param {type:"integer"}

# Universal Plot height
height =  14#@param {type:"integer"}

# Universal xtick size
xtick_size = 8 #@param {type:"integer"}

# Universal ytick size
ytick_size =  8#@param {type:"integer"}

# Universal title font size
title_size = 15 #@param {type:"integer"}

# Universal xlabel font size
xlabel_size = 12 #@param {type:"integer"}

# Universal xlabel font size
ylabel_size = 12 #@param {type:"integer"}

# Universal zlabel font size
zlabel_size = 12 #@param {type:"integer"}

# Universal legend font size
legend_size = 10 #@param {type:"integer"}

# Universal plot font color
color_plots = 'black'

plt.rcParams['figure.figsize'] = (width,height)
params = {'text.color' : 'black',
          'xtick.color' : color_plots,
          'ytick.color' : color_plots,
          'xtick.labelsize' : xtick_size,
          'ytick.labelsize' : ytick_size,
          'legend.loc' : 'upper left',
         }
plt.rcParams.update(params)

"""### Geometric Brownian Path Generator"""

def gbm_paths(S0,K,T,t,r,q,sigma,seed,iterations,timesteps):
  np.random.seed(seed)    
  dt = T/timesteps
  bt = np.random.randn(int(iterations), int(timesteps))
  S = S0 * np.cumprod((np.exp(sigma * np.sqrt(dt) * bt + (r - q - 0.5 * (sigma**2)) * dt)), axis = 1)
  for i in range(0,len(S)):
    S[i][0] = S0
  return S

"""### **Black-Scholes-Merton Theoretical Price**"""

# Call Options:
def bsm_call(S0, K, T, t, r, q, sigma):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  c = np.exp(-r * (T - t)) * (G0 * N(d1) - K * N(d2))
  return c

# Put Options:
def bsm_put(S0, K, T, t, r, q, sigma):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  p = np.exp(-r * (T - t)) * (K * N(-d2) - G0 * N(-d1))
  return p

# Call Options:
def mc_call_arithm(S0, K, T, t, r, q, sigma,iterations,timesteps):
  mc_call_arithm_payoffs = []
  for i in range(1,iterations):
    S = gbm_paths(S0,K,T,t,r,q,sigma,seed,iterations,timesteps)
    S_arithm = np.sum(S)/timesteps
    mc_call_arithm_payoffs.append(np.exp(-r * T) * max(S_arithm - K, 0))
  c = np.mean(mc_call_arithm_payoffs)
  return c

# Put Options:
def mc_put_arithm(S0, K, T, t, r, q, sigma,iterations,timesteps):
  mc_put_arithm_payoffs = []
  for i in range(1,it):
    S = gbm_paths(S0,K,T,t,r,q,sigma,seed,iterations,timesteps)
    S_arithm = np.sum(S)/len(S)
    mc_put_arithm_payoffs.append(np.exp(-r * T) * max(K - S_arithm, 0))
  p = np.mean(mc_put_arithm_payoffs)
  return p

"""### **Monte Carlo Simulator with Geometric Average**"""

# Call Options:
def mc_call_geom(S0, K, T, t, r, q, sigma,iterations,timesteps):
  mc_call_geom_payoffs = []
  for i in range(1,iterations):
    S = gbm_paths(S0,K,T,t,r,q,sigma,seed,iterations,timesteps)
    S_geom_mu = np.exp(np.mean(np.log(S)))
    mc_call_geom_payoffs.append(np.exp(-r * T) * max(S_geom_mu - K, 0))
  c = np.mean(mc_call_geom_payoffs)
  return c

# Put Options:
def mc_put_geom(S0, K, T, t, r, q, sigma,iterations,timesteps):
  mc_put_geom_payoffs = []
  for i in range(1,iterations):
    S = gbm_paths(S0,K,T,t,r,q,sigma,seed,iterations,timesteps)
    S_geom = np.exp(np.mean(np.log(S)))
    mc_put_geom_payoffs.append(np.exp(-r * T) * max(K - S_geom, 0))
  p = np.mean(mc_put_geom_payoffs)
  return p

"""### PDF, CDF"""

# Probability density function of standard normal
def dN(x):
  return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

# Cumulative density function of standard normal
def N(u):
  q = special.erf(u / np.sqrt(2.0))
  return (1.0 + q) / 2.0

"""### **Greeks**"""

# 1st Order Greeks:

# Delta
def bsm_delta(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  if(optiontype == "Call"):
    delta = np.exp(-(T - t)) * N(d1)
  elif(optiontype == "Put"):
    delta = -np.exp(-(T - t)) * N(-d1)
  return delta

# Vega
def bsm_vega(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  vega = S0 * dN(d1) * np.sqrt(T - t)
  return vega

# Rho
def bsm_rho(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)  
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  if(optiontype == "Call"):
    rho = K * (T - t) * np.exp(-r * (T - t)) * N(d2)
  if(optiontype == "Put"):
    rho = -K * (T - t) * np.exp(-r * (T - t)) * N(-d2)
  return rho

# Theta
def bsm_theta(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  if(optiontype == "Call"):
    theta = -(S0 * dN(d1) * Sigma_G / (2 * np.sqrt(T - t)) - r * K * np.exp(-r * (T - t)) * N(d2))
  if(optiontype == "Put"):
    theta = -(S0 * dN(d1) * Sigma_G / (2 * np.sqrt(T - t)) + r * K * np.exp(-r * (T - t)) * N(-d2))
  return theta

# 2nd Order Greeks:

# Gamma
def bsm_gamma(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  gamma = dN(d1) / (S0 * Sigma_G * math.sqrt(T - t))
  return gamma

# Charm
def bsm_charm(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)  
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t)) 
  charm = -dN(d1) * (2 * r * (T-t) - d2 * Sigma_G * np.sqrt(T-t)) / (2 * (T - t) * Sigma_G * np.sqrt(T-t))
  return charm

# Phi
def bsm_phi(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)  
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  phi = 0.01 * T * S0 * np.exp(-q * T) * N(d1)
  return phi

# Vanna
def bsm_vanna(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)  
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  vanna = S0 * np.exp(-q * T) * d2 / Sigma_G * dN(d1) 
  return vanna

# Vomma
def bsm_vomma(S0, K, T, t, r, q, sigma, optiontype):
  G0 = S0 * np.exp(0.5 * (T - t) * (r - q - (sigma**2)/6))
  Sigma_G = sigma/np.sqrt(3)  
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  vomma = -0.01 * np.exp(-q * T) * d2 / Sigma_G * dN(d1)
  return vomma

"""### Option Sensitivity"""

def mc_plot_values(function,S0, K, T, t, r, q, sigma, iterations, timesteps, optiontype):
    fig = plt.figure(figsize=(30,90))
    points = 100

    # Option(K,T) vs. Strike
    fig1 = fig.add_subplot(821)
    klist = np.linspace(K-30, K+30, points)
    vlist = [function(S0, K, T, t, r, q, sigma,iterations,timesteps) for K in klist]
    fig1.plot(klist, vlist)
    fig1.grid()
    fig1.set_title('Monte Carlo %s Option Value vs. Strike' % optiontype, color = color_plots, fontsize = title_size)
    fig1.set_xlabel('Strike $K$', color = color_plots, fontsize = xlabel_size)
    fig1.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. Strike vs. Underlying Price
    klist = np.linspace(K-30, K+30, points)
    s0list = np.linspace(S0 - 20, S0 + 20, points)
    V = np.zeros((len(s0list), len(klist)), dtype=np.float)
    for j in range(len(klist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], klist[j], T, t, r, q, sigma,iterations,timesteps)

    fig2 = fig.add_subplot(823, projection="3d")
    x, y = np.meshgrid(klist, s0list)
    fig2.patch.set_alpha(0.0)
    fig2.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig2.set_title('Monte Carlo %s Option Value vs. Strike vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig2.set_xlabel('Strike $K$', color = color_plots, fontsize = xlabel_size)
    fig2.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig2.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

    # Option(K,T) vs. Time
    fig3 = fig.add_subplot(822)
    tlist = np.linspace(0.0001, T, points)
    vlist = [function(S0, K, T, t, r, q, sigma,iterations,timesteps) for T in tlist]
    fig3.plot(tlist, vlist)
    fig3.grid()
    fig3.set_title('Monte Carlo %s Option Value vs. Time' % optiontype, color = color_plots, fontsize = title_size)
    fig3.set_xlabel('Maturity $T$', color = color_plots, fontsize = xlabel_size)
    fig3.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. Time vs. Underlying Price
    tlist = np.linspace(0.0001, T, points)
    s0list = np.linspace(S0 - 20, S0 + 10, points)
    V = np.zeros((len(s0list), len(tlist)), dtype=np.float)
    for j in range(len(tlist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], K, tlist[j], t, r, q, sigma,iterations,timesteps)

    fig4 = fig.add_subplot(824, projection="3d")
    x, y = np.meshgrid(tlist, s0list)
    fig4.patch.set_alpha(0.0)
    fig4.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig4.set_title('Monte Carlo %s Option Value vs. Time vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig4.set_xlabel('Maturity $T$', color = color_plots, fontsize = xlabel_size)
    fig4.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig4.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

    # Option(K,T) vs. r
    fig5 = fig.add_subplot(825)
    rlist = np.linspace(0, r, points)
    vlist = [function(S0, K, T, t, r, q, sigma,iterations,timesteps) for r in rlist]
    fig5.plot(rlist, vlist)
    fig5.grid()
    fig5.set_title('Monte Carlo %s Option Value vs. r' % optiontype, color = color_plots, fontsize = title_size)
    fig5.set_xlabel('Risk-free rate $r$', color = color_plots, fontsize = xlabel_size)
    fig5.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. r vs. Underlying Price
    rlist = np.linspace(0, r, points)
    s0list = np.linspace(S0 - 20, S0 + 20, points)
    V = np.zeros((len(s0list), len(rlist)), dtype=np.float)
    for j in range(len(rlist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], K, T, t, rlist[j], q, sigma,iterations,timesteps)

    fig6 = fig.add_subplot(827, projection="3d")
    x, y = np.meshgrid(rlist, s0list)
    fig6.patch.set_alpha(0.0)
    fig6.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig6.set_title('Monte Carlo %s Option Value vs. r vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig6.set_xlabel('Risk-free rate $r$', color = color_plots, fontsize = xlabel_size)
    fig6.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig6.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

    # Option(K,T) vs. Implied Vol.
    fig7 = fig.add_subplot(826)
    slist = np.linspace(0.01, sigma, points)
    vlist = [function(S0, K, T, t, r, q, sigma,iterations,timesteps) for sigma in slist]
    fig7.plot(slist, vlist)
    fig7.grid()
    fig7.set_title('Monte Carlo %s Option Value vs. Volatility' % optiontype, color = color_plots, fontsize = title_size)
    fig7.set_xlabel('Volatility $\sigma$', color = color_plots, fontsize = xlabel_size)
    fig7.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. Volatility vs. Underlying Price
    slist = np.linspace(0.01, sigma, points)
    s0list = np.linspace(S0 - 20, S0 + 20, points)
    V = np.zeros((len(s0list), len(slist)), dtype=np.float)
    for j in range(len(slist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], K, T, t, r, q, slist[j],iterations,timesteps)

    fig8 = fig.add_subplot(828, projection="3d")
    x, y = np.meshgrid(slist, s0list)
    fig8.patch.set_alpha(0.0)
    fig8.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig8.set_title('Monte Carlo %s Option Value vs. Volatility vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig8.set_xlabel('Volatility $\sigma$', color = color_plots, fontsize = xlabel_size)
    fig8.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig8.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

"""### **Variance Reduction**"""

# Call Options:
# Call Options:
def mc_call_control_variates(S0, K, T, t, r, q, sigma,iterations,timesteps, simulations):
  c_cv_temp = []
  c_cv = []
  c_upper_cv = []
  c_lower_cv = []
  for i in range(1,simulations):
    seed = random.randint(1,100000)
    S = gbm_paths(S0, K, T, t, r, q, sigma, seed, iterations, timesteps)
    S_arithm = np.mean(S)
    S_geom = np.exp(np.mean(np.log(S)))
    payoff_arithm = np.exp(-r * T) * max(S_arithm - K, 0)
    payoff_geom = np.exp(-r * T) * max(S_geom - K, 0)
    c_cv_temp.append(payoff_arithm - payoff_geom)
    c_cv.append(np.mean(c_cv_temp))
    c_upper_cv.append(np.mean(c_cv) + 1.96 * np.std(c_cv)/np.sqrt(i))
    c_lower_cv.append(np.mean(c_cv) - 1.96 * np.std(c_cv)/np.sqrt(i))
  return c_cv, c_upper_cv, c_lower_cv

# Put Options:
def mc_put_control_variates(S0, K, T, t, r, q, sigma,iterations,timesteps,simulations):
  p_cv_temp = []
  p_cv = []
  p_upper_cv = []
  p_lower_cv = []
  for i in range(1,simulations):
    seed = random.randint(1,100000)
    S = gbm_paths(S0, K, T, r, q, sigma, seed, iterations, timesteps)
    S_arithm = np.mean(S)
    S_geom = np.exp(np.mean(np.log(S)))
    payoff_arithm = np.exp(-r * T) * max(K - S_arithm, 0)
    payoff_geom = np.exp(-r * T) * max(K - S_geom, 0)
    p_cv.append(payoff_geom - payoff_arithm)
    p_cv.append(np.mean(p_cv_temp))
    p_upper_cv.append(np.mean(p_cv) + 1.96 * np.std(p_cv)/np.sqrt(i))
    p_lower_cv.append(np.mean(p_cv) - 1.96 * np.std(p_cv)/np.sqrt(i))
  return p_cv, p_upper_cv, p_lower_cv
"""### Multiple Control Variates"""

# Call Options:(S0,K,T,t,r,q,sigma,seed,iterations,timesteps):
mc_call_cv = mc_call_control_variates(100, 95, 1.0, 0.0, 0.15, 0.0, 0.3, 10000, 100000,1000)
c_cv_values = mc_call_cv[0]
c_cv_upper = mc_call_cv[1]c
c_cvv_lower = mc_call_cv[2]
data_call = [c_cv_upper, c_cvv_lower, c_cv_values]
for i in range(0,len(data_call)):
  plt.plot(data_call[i])
plt.title('$Arithmetic - Geometric Call Option Value Difference',color=color_plots,fontsize=title_size)
plt.xlabel('$\Delta$t', color=color_plots, fontsize=xlabel_size)
plt.ylabel('Difference', color=color_plots, fontsize=ylabel_size)
plt.show()

# # Put Options:
# mc_put_cv = mc_put_control_variates(100, 95, 1.0, 0.0, 0.15, 0.0, 0.3, 1000, 4000)
# p_cv_values = mc_put_cv[0]
# p_cv_upper = mc_put_cv[1]
# p_cvv_lower = mc_put_cv[2]
# data_put = [p_cv_upper, p_cvv_lower, p_cv_values]
# for i in range(0,len(data_put)):
#   plt.plot(data_put[i])
# plt.title('$Arithmetic - Geometric Call Option Value Difference',color=color_plots,fontsize=title_size)
# plt.xlabel('$\Delta$t', color=color_plots, fontsize=xlabel_size)
# plt.ylabel('Difference', color=color_plots, fontsize=ylabel_size)
# plt.show()

"""## **Hedging Test**"""

daily_data = np.ones((10,timesteps))
S_geometric = np.ones((10,1))
payoffs = np.ones((10,1))
for i in range(0,10):
  seed = np.random.randint(1,5000)
  daily_data[i] = gbm_paths(100,95,1.0,0.0,0.0,0.15,0.3,seed,1,timesteps)

plt.figure(figsize=(width,height))
_= plt.plot(np.transpose(daily_data))
_ = plt.title('Simulated Stock Prices',fontsize=title_size,color = color_plots)
_ = plt.ylabel('Price',fontsize=ylabel_size,color = color_plots)
_ = plt.xlabel('Time Step',fontsize=xlabel_size,color = color_plots)

plt.figure(figsize=(width,height))
_= plt.plot(payoffs)
_ = plt.title('Payouts',fontsize=title_size,color = color_plots)
_ = plt.ylabel('Price',fontsize=ylabel_size,color = color_plots)
_ = plt.xlabel('Time Step',fontsize=xlabel_size,color = color_plots)

print(payoffs)
print(daily_data)
#
# """### Portfolio Sensitivity
# **1st Order BSM Greeks: Sensitivity to Strike and Stock/Underlying Price**
# """
#
# fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(30, 30))
# #Variables:
# klist = [80,95,100,105,120]
# S0list = np.arange(50,150)
# r = 0.15
# sigma = 0.3
# T = 1
# t = 0.0
# q=0.0
#
# plt.subplot(321)
# for i in klist:
#     c = [bsm_delta(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
#     p = [bsm_delta(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
#     plt.plot(c, label = ("Delta Call K=%i" % i ))
#     plt.plot(p, label = ("Delta Put K=%i" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Delta", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(322)
# for i in klist:
#     c = [bsm_gamma(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
#     p = [bsm_gamma(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
#     plt.plot(c, label = ("Gamma Call K=%i" % i ))
#     plt.plot(p, label = ("Gamma Put K=%i" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Gamma", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(323)
# for i in klist:
#     c = [bsm_vega(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
#     p = [bsm_vega(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
#     plt.plot(c, label = ("Vega Call K=%i" % i ))
#     plt.plot(p, label = ("Vega Put K=%i" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Vega", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(324)
# for i in klist:
#     c = [bsm_rho(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
#     p = [bsm_rho(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
#     plt.plot(c, label = ("Rho Call K=%i" % i ))
#     plt.plot(p, label = ("Rho Put K=%i" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Rho", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(325)
# for i in klist:
#     c = [bsm_theta(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
#     p = [bsm_theta(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
#     plt.plot(c, label = ("Theta Call K=%i" % i ))
#     plt.plot(p, label = ("Theta Put K=%i" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Theta", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(326)
# for i in klist:
#     c = [bsm_charm(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
#     p = [bsm_charm(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
#     plt.plot(c, label = ("Charm Call K=%i" % i ))
#     plt.plot(p, label = ("Charm Put K=%i" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Charm", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
# plt.show()
#
# """**1st Order BSM Greeks: Sensitivity to Risk-Free Rate + Stock/Underlying Price**"""
#
# fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(30, 30))
#
# # Variables:
# rlist = [0.0,0.05,0.1,0.15,0.2]
# S0list = np.arange(50,150)
# K = 95
# r = 0.15
# sigma = 0.3
# T = 1.0
# t = 0.0
# q = 0.0
#
# plt.subplot(321)
# for i in rlist:
#   c = [bsm_delta(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
#   p = [bsm_delta(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
#   plt.plot(c, label = ("Delta Call r=%.2f" % i ))
#   plt.plot(p, label = ("Delta Put r=%.2f" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Delta", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(322)
# for i in rlist:
#   c = [bsm_gamma(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
#   p = [bsm_gamma(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
#   plt.plot(c, label = ("Gamma Call r=%.2f" % i ))
#   plt.plot(p, label = ("Gamma Put r=%.2f" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Gamma", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(323)
# for i in rlist:
#   c = [bsm_vega(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
#   p = [bsm_vega(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
#   plt.plot(c, label = ("Vega Call r=%.2f" % i ))
#   plt.plot(p, label = ("Vega Put r=%.2f" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Vega", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(324)
# for i in rlist:
#   c = [bsm_rho(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
#   p = [bsm_rho(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
#   plt.plot(c, label = ("Rho Call r=%.2f" % i ))
#   plt.plot(p, label = ("Rho Put r=%.2f" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Rho", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(325)
# for i in rlist:
#   c = [bsm_theta(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
#   p = [bsm_theta(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
#   plt.plot(c, label = ("Theta Call r=%.2f" % i ))
#   plt.plot(p, label = ("Theta Put r=%.2f" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Theta", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
#
# plt.subplot(326)
# for i in rlist:
#   c = [bsm_charm(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
#   p = [bsm_charm(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
#   plt.plot(c, label = ("Charm Call r=%.2f" % i ))
#   plt.plot(p, label = ("Charm Put r=%.2f" % i ))
#
# plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
# plt.ylabel("Charm", color = color_plots, fontsize = ylabel_size)
# plt.legend(fontsize = legend_size)
# plt.show()