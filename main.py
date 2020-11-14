# -*- coding: utf-8 -*-

## **Comments**

**Instructions:**

* Click the execution button on the top left corner of each code cell to execute it. To run all cells in descending order, go to the menu bar at the top and click Runtime -> Run all (or use the **Ctrl-F9** or **⌘-F9** hotkey for Windows and MacOSX respectively);
* If the code is running slowly, go to Runtime -> Change runtime type, and change the Runtime Shape to High-RAM from the dropdown menu;
* The **"!pip install"** lines under the Python packages section (i.e. lines 3-5) only need to be executed the first time you run the notebook. If you receive the message **"Requirement already satisfied:"**, wrap them in treble quotes (add the quotes in lines 2 & 6)*.

## **Python Packages**
"""

#Installations#
'''
!pip install -q quandl
!pip install -q yfinance
!pip install quantsbin
!pip install quantumRandom
'''

# Main modules
import math
import numpy as np
import pandas as pd
'''
import quandl
import yfinance
import quantsbin
'''
# Scipy
from scipy import stats
from scipy.stats import norm
from scipy.integrate import quad
from scipy import special

# Plotters
import matplotlib as mpl
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# Helper modules
import timeit
import datetime
import random
import os
import requests
from datetime import timedelta
#import quantumrandom

# High-performance computing
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
'''
import jax
import jax.numpy as jnp
from jax.config import config
from jax.ops import index, index_add, index_update
import numba as nb
from numba import jit, njit, vectorize, cuda

# GPU Pointers for Numba
#os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-10.1/nvvm/libdevice"
#os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-10.1/nvvm/lib64/libnvvm.so"

# TPU Pointers for JAX
if 'TPU_DRIVER_MODE' not in globals():
  url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver_nightly'
  resp = requests.post(url)
  TPU_DRIVER_MODE = 1
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
print(config.FLAGS.jax_backend_target)

"""## **Variables**

### Model Variables
"""

# Initial Underlying Price
S0 = 100                                                #@param {type:"number" }
# Risk-free rate
r = 0.15                                                #@param {type:"number" }
# Dividend Yield Rate
q = 0.0                                                 #@param {type:"number" }
# Valuation Date
t = 0.0                                                 #@param {type:"number" }
# Maturity
T = 1.0                                                 #@param {type:"number" }
# Strike
K = 100                                                 #@param {type:"number" }
# Volatility
sigma = 0.3                                             #@param {type:"number" }
# Number of Price paths for each Monte Carlo simulation
n_paths = 100                                           #@param {type:"integer"}
# Time Steps
n_steps = 504                                           #@param {type:"integer"}
# Number of Monte Carlo simulations
n_sims = 1000                                           #@param {type:"integer"}
# Initial Seed
seed_0 = 42                                             #@param {type:"integer"}
# Initial Seed
seed = 100                                              #@param {type:"integer"}

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

"""## **Definitions**

### Brownian Path Generator
"""

def bm_paths(n_paths,n_sims,seed):
  if seed != 0:
    np.random.seed(seed) 
  dt = T / n_steps
  bt = np.random.randn(int(n_paths), int(n_sims))
  W = np.cumprod(bt,axis=1)
  return W

"""### Geometric Brownian Path Generator

#### Conventional Implementation:
"""

def gbm_paths_original(S0,K,T,t,r,q,sigma,seed,n_paths,n_steps):
  np.random.seed(seed)    
  dt = T/n_steps
  bt = np.random.randn(int(n_paths), int(n_steps))
  S = S0 * np.cumprod((np.exp(sigma * np.sqrt(dt) * bt + (r - q - 0.5 * (sigma**2)) * dt)), axis = 1)
  for i in range(0,len(S)):
    S[i][0] = S0
  return S

"""#### JAX Implementation"""

def gbm_paths_jax(S0, r, q, sigma, n_steps,n_paths):
  dt = T / n_steps
  seed = np.random.randint(0,1000000)
  S = jnp.exp((r - q - sigma ** 2 / 2) * dt + np.sqrt(dt) * sigma * jax.random.normal(jax.random.PRNGKey(seed), (n_steps, n_paths)))
  S = jnp.vstack([np.ones(n_paths), S])
  S = S0 * jnp.cumprod(S, axis=0)
  return S

def gbm_paths_jax_iter(S0, r, q, sigma, n_steps,n_paths):
  seed = np.random.randint(0,1000000)
  key = jax.random.PRNGKey(seed)
  dt = T / n_steps
  S = np.zeros((n_steps + 1, n_paths))
  S[0] = S0
  rn = jax.random.normal(key, shape=S.shape)
  for t in range(1, n_steps + 1): 
    S[t] = S[t-1] * np.exp((r - q - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rn[t])
    print(S[t])
  return S



"""#### Tensorflow Implementation:"""

def tf_graph_gbm_paths():
  S0 = tf.placeholder(tf.float32)
  K = tf.placeholder(tf.float32)
  dt = tf.placeholder(tf.float32)
  T = tf.placeholder(tf.float32)
  sigma = tf.placeholder(tf.float32)
  r = tf.placeholder(tf.float32)
  dw = tf.placeholder(tf.float32)
  S_i = S0 * tf.cumprod(tf.exp((r-sigma**2/2)*dt+sigma*tf.sqrt(dt)*dw), axis=1)
  return (S0, K, dt, T, sigma, r, dw, S_i)

def tf_gbm_paths():
  (S0,K, dt, T, sigma, r, dw, S_i) = tf_graph_gbm_paths()
  def paths(S_zero, strk, maturity, riskfrate, volatility, seed, n_paths, n_steps):
    if seed != 0:
      np.random.seed(seed)
    stdnorm_random_variates = np.random.randn(n_paths, n_steps)
    with tf.Session() as sess:
      delta_t = maturity / n_steps
      res = sess.run(S_i,
                     {
                         S0: S_zero,
                         K : strk,
                         r : riskfrate,
                         sigma: volatility,
                         dt : delta_t,
                         T: maturity,
                         dw : stdnorm_random_variates})
      return res
  return paths

"""### **Black-Scholes-Merton Theoretical Price**
Call and Put Asian option prices with geometric averaging. The alternative implementation in Tensorlfow offers efficiency and includes the calculations of 1st 2nd and 3rd order Greeks.

#### Conventional Implementation
"""

# Call Options:
def bsm_call(S0, K, T, t, r, q, sigma):
  G0 = S0 * np.exp(0.5 * (r - q) * (T - t) - ((T-t) * (sigma**2))/12)
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  c = np.exp(-r * (T - t)) * (G0 * N(d1) - K * N(d2))
  return c

# Put Options:
def bsm_put(S0, K, T, t, r, q, sigma):
  G0 = S0 * np.exp(0.5 * (r - q) * (T - t) - ((T-t) * (sigma**2))/12)
  Sigma_G = sigma/np.sqrt(3)
  d1 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) + 0.5 * (Sigma_G**2) * (T - t))
  d2 = (1/(Sigma_G * np.sqrt(T - t))) * (np.log(G0/K) - 0.5 * (Sigma_G**2) * (T - t))
  p = np.exp(-r * (T - t)) * (K * N(-d2) - G0 * N(-d1))
  return p

"""#### Tensorflow Implementation"""

# Sample Output
# [NET PRESENT VALUE, [DELTA, VEGA, THETA], [GAMMA, VANNA, CHARM], [VANNA, VOLGA, VETA], [SPEED, ZOMMA, COLOR], [N/A, ULTIMA, TOTTO]]

# Call Options:
def bsm_call_tf(enable_greeks = True):
    S0 = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    dt = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    q = tf.placeholder(tf.float32)    
    G0 = S0 * tf.exp(0.5 * (r * dt) - ((tf.np.square(sigma)) * dt)/12)
    Sigma_G = sigma/tf.sqrt(3.0)
    Phi = tf.distributions.Normal(0.,1.).cdf
    d_1 = (1/(Sigma_G * tf.sqrt(dt))) * (tf.log(G0/K) + 0.5 * (tf.math.square(Sigma_G)) * dt)
    d_2 = (1/(Sigma_G * tf.sqrt(dt))) * (tf.log(G0/K) - 0.5 * (tf.math.square(Sigma_G)) * dt)
    npv =  tf.exp(-r * dt) * (G0 * Phi(d_1) - K * Phi(d_2))                # GREEKS TABLE:
    target_calc = [npv]                                                    # (e.g. Option Price with respect to Asset Price (S) is delta)
    if enable_greeks:                                                      #                Asset Price (S)   Volatility    Time to Expiry
      greeks = tf.gradients(npv, [S0, sigma, dt])                          # Option Price |     delta            vega           theta
      dS_2nd = tf.gradients(greeks[0], [S0, sigma, dt])                    # Delta        |     gamma            vanna          charm
      dsigma_2nd = tf.gradients(greeks[1], [S0, sigma, dt])                # Vega         |     vanna         vomma/volga       veta
      dT_2nd = tf.gradients(dS_2nd[0], [S0, sigma, dt])                    # Gamma        |     speed            zomma          color
      dsigma_3rd = tf.gradients(dsigma_2nd[1], [S0, sigma, dt])            # Vomma        |      N/A             ultima         totto
      target_calc += [greeks, dS_2nd, dsigma_2nd, dT_2nd, dsigma_3rd]
    def execute_graph(S_zero, strk, maturity, riskfrate, volatility):
        with tf.Session() as sess:
            res = sess.run(target_calc, 
                           {
                               S0: S_zero,
                               K : strk,
                               r : riskfrate,
                               sigma: volatility,
                               dt: maturity})
        return res
    return execute_graph

# Put Options:
def bsm_put_tf(enable_greeks = True):
    S0 = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    dt = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    q = tf.placeholder(tf.float32)    
    G0 = S0 * tf.exp(0.5 * (r * dt) - ((tf.math.square(sigma)) * dt)/12)
    Sigma_G = sigma/tf.sqrt(3.0)
    Phi = tf.distributions.Normal(0.,1.).cdf
    d_1 = (1/(Sigma_G * tf.sqrt(dt))) * (tf.log(G0/K) + 0.5 * (tf.math.square(Sigma_G)) * dt)
    d_2 = (1/(Sigma_G * tf.sqrt(dt))) * (tf.log(G0/K) - 0.5 * (tf.math.square(Sigma_G)) * dt)
    npv =  tf.exp(-r * dt) * (K * Phi(-d_2) - G0 * Phi(-d_1))              # GREEKS TABLE:
    target_calc = [npv]                                                    # (e.g. Option Price with respect to Asset Price (S) is delta)
    if enable_greeks:                                                      #                Asset Price (S)   Volatility    Time to Expiry
      greeks = tf.gradients(npv, [S0, sigma, dt])                          # Option Price |     delta            vega           theta
      dS_2nd = tf.gradients(greeks[0], [S0, sigma, dt])                    # Delta        |     gamma            vanna          charm
      dsigma_2nd = tf.gradients(greeks[1], [S0, sigma, dt])                # Vega         |     vanna         vomma/volga       veta
      dT_2nd = tf.gradients(dS_2nd[0], [S0, sigma, dt])                    # Gamma        |     speed            zomma          color
      dsigma_3rd = tf.gradients(dsigma_2nd[1], [S0, sigma, dt])            # Vomma        |      N/A             ultima         totto
      target_calc += [greeks, dS_2nd, dsigma_2nd, dT_2nd, dsigma_3rd]
    def execute_graph(S_zero, strk, maturity, riskfrate, volatility):
        with tf.Session() as sess:
            res = sess.run(target_calc, 
                           {
                               S0: S_zero,
                               K : strk,
                               r : riskfrate,
                               sigma: volatility,
                               dt: maturity})
        return res
    return execute_graph

"""### **Monte Carlo Simulator with Arithmetic Average**
Call and Put Asian option prices with arithmetic averaging. The alternative implementation in Tensorlfow offers efficiency and includes the calculations of 1st 2nd and 3rd order Greeks.

#### Conventional Interpretation
"""

# Call Options:
def mc_call_arithm_np(S0, K, T, t, r, q, sigma, seed, n_paths, n_steps):
  mc_call_arithm_payoffs = []
  for i in range(1,n_paths):
    seed = np.random.randint(1,2**31)
    S = gbm_paths(S0,K,T,t,r,q,sigma,seed,n_paths,n_steps)
    S_arithm_mu = np.mean(S)
    arithm_payoff_call = np.exp(-r * T) * max(S_arithm_mu - K, 0)
    mc_call_arithm_payoffs.append(arithm_payoff_call)
  c = np.mean(mc_call_arithm_payoffs)
  return c

# Put Options:
def mc_put_arithm_np(S0, K, T, t, r, q, sigma, seed, n_paths, n_steps):
  mc_put_arithm_payoffs = []
  for i in range(1,n_paths):
    seed = np.random.randint(1,2**31)
    S = gbm_paths(S0,K,T,t,r,q,sigma,seed,n_paths,n_steps)
    S_arithm_mu = np.mean(S)
    arithm_payoff_put = np.exp(-r * T) * max(K - S_arithm_mu, 0)
    print(arithm_payoff_put)
    mc_put_arithm_payoffs.append(arithm_payoff_put)
  p = np.mean(mc_put_arithm_payoffs)
  return p

"""#### JAX Implementation"""

def mc_call_arithm(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims):
  c = []
  K = np.full(n_paths,K)
  payoff_0 = np.zeros(n_paths)
  for i in range(1,n_sims):
    S = gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)
    S_arithm = jnp.mean(S, axis=0)
    c.append(jnp.mean(jnp.exp(-r*T) * jnp.maximum(S_arithm - K, payoff_0)))
  call = np.mean(c)
  return call

def mc_put_arithm(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims):
  p = []
  K = np.full(n_paths,K)
  payoff_0 = np.zeros(n_paths)
  for i in range(1,n_sims):
    S = gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)
    S_arithm = jnp.mean(S, axis=0)
    p.append(jnp.mean(jnp.exp(-r*T) * jnp.maximum(K - S_arithm, payoff_0)))
  put = np.mean(p)
  return put

"""#### Tensorflow Implementation:"""

# Sample Output
# [NET PRESENT VALUE, [DELTA, VEGA, THETA], [GAMMA, VANNA, CHARM], [VANNA, VOLGA, VETA], [SPEED, ZOMMA, COLOR], [N/A, ULTIMA, TOTTO]]

# Call Options:
def mc_call_arithm_tf(enable_greeks=False):
    (S0, K, dt, T, sigma, r, dw, S_i) = tf_graph_gbm_paths()
    A = tf.reduce_sum(S_i, axis=1)/(T/dt)
    payout = tf.maximum(A - K, 0)
    npv = tf.exp(-r * T) * tf.reduce_mean(payout)
    target_calc = [npv]
    if enable_greeks:
      greeks = tf.gradients(npv, [S0, sigma, dt]) # delta, vega, theta
      dS_2nd = tf.gradients(greeks[0], [S0, sigma, dt]) # gamma, vanna, charm
      dsigma_2nd = tf.gradients(greeks[1], [S0, sigma, dt]) # vanna, vomma/volga, veta
      dT_2nd = tf.gradients(dS_2nd[0], [S0, sigma, dt]) # speed, zomma, color
      dsigma_3rd = tf.gradients(dsigma_2nd[1], [S0, sigma, dt]) # N/A, ultima, totto
      target_calc += [greeks, dS_2nd, dsigma_2nd, dT_2nd, dsigma_3rd]
    def pricer(S_zero, strk, maturity, volatility, riskfrate, seed, n_paths, n_steps):
      if seed != 0:
        np.random.seed(seed)
      stdnorm_random_variates = np.random.randn(n_paths, n_steps)
      with tf.Session() as sess:
        delta_t = maturity / n_steps
        res = sess.run(target_calc,
                       {
                           S0: S_zero,
                           K: strk,
                           r: riskfrate,
                           sigma: volatility,
                           dt: delta_t,
                           T: maturity,
                           dw: stdnorm_random_variates})
        return res
    return pricer


# Put Options:
def mc_put_arithm_tf(enable_greeks=False):
    (S0, K, dt, T, sigma, r, dw, S_i) = tf_graph_gbm_paths()
    A = tf.reduce_sum(S_i, axis=1)/(T/dt)
    payout = tf.maximum(K - A, 0)
    npv = tf.exp(-r * T) * tf.reduce_mean(payout)
    target_calc = [npv]
    if enable_greeks:
      greeks = tf.gradients(npv, [S0, sigma, dt]) # delta, vega, theta
      dS_2nd = tf.gradients(greeks[0], [S0, sigma, dt]) # gamma, vanna, charm
      dsigma_2nd = tf.gradients(greeks[1], [S0, sigma, dt]) # vanna, vomma/volga, veta
      dT_2nd = tf.gradients(dS_2nd[0], [S0, sigma, dt]) # speed, zomma, color
      dsigma_3rd = tf.gradients(dsigma_2nd[1], [S0, sigma, dt]) # N/A, ultima, totto
      target_calc += [greeks, dS_2nd, dsigma_2nd, dT_2nd, dsigma_3rd]
    def pricer(S_zero, strk, maturity, volatility, riskfrate, seed, n_paths, n_steps):
      if seed != 0:
        np.random.seed(seed)
      stdnorm_random_variates = np.random.randn(n_paths, n_steps)
      with tf.Session() as sess:
        delta_t = maturity / n_steps
        res = sess.run(target_calc,
                       {
                           S0: S_zero,
                           K: strk,
                           r: riskfrate,
                           sigma: volatility,
                           dt: delta_t,
                           T: maturity,
                           dw: stdnorm_random_variates})
        return res
    return pricer

"""### **Monte Carlo Simulator with Geometric Average**

#### Conventional Implementation
"""

# Call Options:
def mc_call_geom_np(S0, K, T, t, r, q, sigma,n_paths,n_steps,n_sims):
  mc_call_geom_payoffs = []
  for i in range(1,n_sims):
    S = gbm_paths_original(S0,K,T,t,r,q,sigma,seed,n_paths,n_steps)
    S_geom_mu = np.exp(np.mean(np.log(S)))
    mc_call_geom_payoffs.append(np.exp(-r * T) * max(S_geom_mu - K, 0))
  c = np.mean(mc_call_geom_payoffs)
  return c

# Put Options:
def mc_put_geom_np(S0, K, T, t, r, q, sigma,n_paths,n_steps,n_sims):
  mc_put_geom_payoffs = []
  for i in range(1,n_sims):
    S = gbm_paths_original(S0,K,T,t,r,q,sigma,seed,n_paths,n_steps)
    S_geom = np.exp(np.mean(np.log(S)))
    mc_put_geom_payoffs.append(np.exp(-r * T) * max(K - S_geom, 0))
  p = np.mean(mc_put_geom_payoffs)
  return p

"""#### JAX Implementation"""

def mc_call_geom(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims):
  c = []
  K = np.full(n_paths,K)
  payoff_0 = np.zeros(n_paths)
  for i in range(1,n_sims):
    S = gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)
    S_geom = jnp.exp(jnp.mean(jnp.log(S), axis=0))
    c.append(jnp.mean(jnp.exp(-r*T) * jnp.maximum(S_geom - K, payoff_0)))
  call = np.mean(c)
  return call

def mc_put_geom(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims):
  p = []
  K = np.full(n_paths,K)
  payoff_0 = np.zeros(n_paths)
  for i in range(1,n_sims):
    S = gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)
    S_geom = jnp.exp(jnp.mean(jnp.log(S), axis=0))
    p.append(jnp.mean(jnp.exp(-r*T) * jnp.maximum(K - S_geom, payoff_0)))
  put = np.mean(p)
  return put

"""#### Tensorflow Implementation"""

# Sample Output
# [NET PRESENT VALUE, [DELTA, VEGA, THETA], [GAMMA, VANNA, CHARM], [VANNA, VOLGA, VETA], [SPEED, ZOMMA, COLOR], [N/A, ULTIMA, TOTTO]]

# Call Options:
def mc_call_geom_tf(enable_greeks=True):
    (S0, K, dt, T, sigma, r, dw, S_i) = tf_graph_gbm_paths()
    A = tf.pow(tf.reduce_prod(S_i, axis=1), dt / T)
    payout = tf.maximum(A - K, 0)
    npv = tf.exp(-r * T) * tf.reduce_mean(payout)                          # GREEKS TABLE:
    target_calc = [npv]                                                    # (e.g. Option Price with respect to Asset Price (S) is delta)
    if enable_greeks:                                                      #
      grads_greeks = tf.ones([n_paths,1])                               #                Asset Price (S)   Volatility    Time to Expiry
      greeks = tf.gradients(npv, [S0, sigma, dt])                          # Option Price |     delta            vega           theta
      dS_2nd = tf.gradients(greeks[0], [S0, sigma, dt])                    # Delta        |     gamma            vanna          charm
      dsigma_2nd = tf.gradients(greeks[1], [S0, sigma, dt])                # Vega         |     vanna         vomma/volga       veta
      dT_2nd = tf.gradients(dS_2nd[0], [S0, sigma, dt])                    # Gamma        |     speed            zomma          color
      dsigma_3rd = tf.gradients(dsigma_2nd[1], [S0, sigma, dt])            # Vomma        |      N/A             ultima         totto
      target_calc += [greeks, dS_2nd, dsigma_2nd, dT_2nd, dsigma_3rd]
    def pricer(S_zero, strk, maturity, riskfrate, volatility, seed, n_paths, n_steps):
      if seed != 0:
        np.random.seed(seed)
      stdnorm_random_variates = np.random.randn(n_paths, n_steps)
      with tf.Session() as sess:
        delta_t = maturity / n_steps
        res = sess.run(target_calc,
                       {
                           S0: S_zero,
                           K: strk,
                           r: riskfrate,
                           sigma: volatility,
                           dt: delta_t,
                           T: maturity,
                           dw: stdnorm_random_variates})
        return res
    return pricer

# Put Options:
def mc_put_geom_tf(enable_greeks=True):
    (S0, K, dt, T, sigma, r, dw, S_i) = tf_graph_gbm_paths()
    A = tf.pow(tf.reduce_prod(S_i, axis=1), dt / T)
    payout = tf.maximum(K - A, 0)
    npv = tf.exp(-r * T) * tf.reduce_mean(payout)                          # GREEKS TABLE:
    target_calc = [npv]                                                    # (e.g. Option Price with respect to Asset Price (S) is delta)
    if enable_greeks:                                                      #                Asset Price (S)   Volatility    Time to Expiry
      greeks = tf.gradients(npv, [S0, sigma, dt])                          # Option Price |     delta            vega           theta
      dS_2nd = tf.gradients(greeks[0], [S0, sigma, dt])                    # Delta        |     gamma            vanna          charm
      dsigma_2nd = tf.gradients(greeks[1], [S0, sigma, dt])                # Vega         |     vanna         vomma/volga       veta
      dT_2nd = tf.gradients(dS_2nd[0], [S0, sigma, dt])                    # Gamma        |     speed            zomma          color
      dsigma_3rd = tf.gradients(dsigma_2nd[1], [S0, sigma, dt])            # Vomma        |      N/A             ultima         totto
      target_calc += [greeks, dS_2nd, dsigma_2nd, dT_2nd, dsigma_3rd]
    def pricer(S_zero, strk, maturity, riskfrate, volatility, seed, n_paths, n_steps):
      if seed != 0:
        np.random.seed(seed)
      stdnorm_random_variates = np.random.randn(n_paths, n_steps)
      with tf.Session() as sess:
        delta_t = maturity / n_steps
        res = sess.run(target_calc,
                       {
                           S0: S_zero,
                           K: strk,
                           r: riskfrate,
                           sigma: volatility,
                           dt: delta_t,
                           T: maturity,
                           dw: stdnorm_random_variates})
        return res
    return pricer

"""### PDF, CDF"""

# Probability density function of standard normal
def dN(x):
  return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

# Cumulative density function of standard normal
def N(u):
  q = special.erf(u / np.sqrt(2.0))
  return (1.0 + q) / 2.0

"""### **Greeks**

#### Greeks Derivations
"""

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

"""#### Greeks: Sensitivity Plotter"""

def greeks_plot_tool(greek_function, x_var_name, y_var_name,S0, K, T, t, r, q, sigma, x, y, optiontype, greek, plot):
    # Initialise vector to store our option values and then iterate over
    # Assumption that we're using a constant sized vector length for each variable
    # Need to change the variables being iterated over here for each update (possibly a better way to do this)
    V = np.zeros((len(S0), len(S0)), dtype=np.float)
    for i in range(len(S0)):
      for j in range(len(S0)):
        V[i, j] = greek_function(S0[i], K[i], T[j], t[i], r[i], q[i], sigma[i], optiontype)
 
    # Initiliase plotting canvas 
    surf = plot.plot_surface(x, y, V, rstride=1, cstride=1, alpha=0.7, cmap=cm.RdYlGn, antialiased = True) #cmap = cm.RdYlBu, for surface plotting
    plot.set_xlabel(x_var_name, color = color_plots, fontsize = xlabel_size)
    plot.set_ylabel(y_var_name, color = color_plots, fontsize = ylabel_size)
    plot.set_zlabel("%s(K, T)" % greek, color = color_plots, fontsize = zlabel_size)
    plot.set_title("%s %s" % (optiontype, greek), color = color_plots, fontsize=title_size, fontweight='bold')
    
    # Calculate colour levels based on our meshgrid
    Vlevels = np.linspace(V.min(), V.max(), num=8, endpoint=True)
    xlevels = np.linspace(x.min(), x.max(), num=8, endpoint=True)
    ylevels = np.linspace(y.min(), y.max(), num=8, endpoint=True)
    
    cset = plot.contourf(x, y, V, Vlevels, zdir='z',offset=V.min(), cmap=cm.RdYlGn, alpha = 0.5, antialiased = True)
    cset = plot.contourf(x, y, V, xlevels, zdir='x',offset=x.min(), cmap=cm.RdYlGn, alpha = 0.5, antialiased = True)
    cset = plot.contourf(x, y, V, ylevels, zdir='y',offset=y.max(), cmap=cm.RdYlGn, alpha = 0.5, antialiased = True)

    # Set our viewing constraints
    plt.clabel(cset,fontsize=10, inline=1, color = color_plots, antialiased = True)
    plot.set_xlim(x.min(), x.max())
    plot.set_ylim(y.min(), y.max())
    plot.set_zlim(V.min(), V.max())

    # Colorbar
    colbar = plt.colorbar(surf, shrink=1.0, extend='both', aspect = 10)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = colbar.ax.get_position().bounds
    colbar.ax.set_position([ll, b+0.1*h, ww, h*0.8])

"""### Option Sensitivity

#### BSM Option Sensitivity
"""

def bsm_plot_values(function,S0, K, T, t, r, q, sigma,optiontype):
    fig = plt.figure(figsize=(30,90))
    points = 100

    # Option(K,T) vs. Strike
    fig1 = fig.add_subplot(821)
    klist = np.linspace(K-30, K+30, points)
    vlist = [function(S0, K, T, t, r, q, sigma) for K in klist]
    fig1.plot(klist, vlist)
    fig1.grid()
    fig1.set_title('BSM %s Option Value vs. Strike' % optiontype, color = color_plots, fontsize = title_size)
    fig1.set_xlabel('Strike $K$', color = color_plots, fontsize = xlabel_size)
    fig1.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. Strike vs. Underlying Price
    klist = np.linspace(K-30, K+30, points)
    s0list = np.linspace(S0 - 20, S0 + 20, points)
    V = np.zeros((len(s0list), len(klist)), dtype=np.float)
    for j in range(len(klist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], klist[j], T, t, r, q, sigma)

    fig2 = fig.add_subplot(823, projection="3d")
    x, y = np.meshgrid(klist, s0list)
    fig2.patch.set_alpha(0.0)
    fig2.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig2.set_title('BSM %s Option Value vs. Strike vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig2.set_xlabel('Strike $K$', color = color_plots, fontsize = xlabel_size)
    fig2.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig2.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

    # Option(K,T) vs. Time
    fig3 = fig.add_subplot(822)
    tlist = np.linspace(0.0001, T, points)
    vlist = [function(S0, K, T, t, r, q, sigma) for T in tlist]
    fig3.plot(tlist, vlist)
    fig3.grid()
    fig3.set_title('BSM %s Option Value vs. Time' % optiontype, color = color_plots, fontsize = title_size)
    fig3.set_xlabel('Maturity $T$', color = color_plots, fontsize = xlabel_size)
    fig3.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. Time vs. Underlying Price
    tlist = np.linspace(0.0001, T, points)
    s0list = np.linspace(S0 - 20, S0 + 10, points)
    V = np.zeros((len(s0list), len(tlist)), dtype=np.float)
    for j in range(len(tlist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], K, tlist[j], t, r, q, sigma)

    fig4 = fig.add_subplot(824, projection="3d")
    x, y = np.meshgrid(tlist, s0list)
    fig4.patch.set_alpha(0.0)
    fig4.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig4.set_title('BSM %s Option Value vs. Time vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig4.set_xlabel('Maturity $T$', color = color_plots, fontsize = xlabel_size)
    fig4.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig4.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

    # Option(K,T) vs. r
    fig5 = fig.add_subplot(825)
    rlist = np.linspace(0, r, points)
    vlist = [function(S0, K, T, t, r, q, sigma) for r in rlist]
    fig5.plot(rlist, vlist)
    fig5.grid()
    fig5.set_title('BSM %s Option Value vs. r' % optiontype, color = color_plots, fontsize = title_size)
    fig5.set_xlabel('Risk-free rate $r$', color = color_plots, fontsize = xlabel_size)
    fig5.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. r vs. Underlying Price
    rlist = np.linspace(0, r, points)
    s0list = np.linspace(S0 - 20, S0 + 20, points)
    V = np.zeros((len(s0list), len(rlist)), dtype=np.float)
    for j in range(len(rlist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], K, T, t, rlist[j], q, sigma)

    fig6 = fig.add_subplot(827, projection="3d")
    x, y = np.meshgrid(rlist, s0list)
    fig6.patch.set_alpha(0.0)
    fig6.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig6.set_title('BSM %s Option Value vs. r vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig6.set_xlabel('Risk-free rate $r$', color = color_plots, fontsize = xlabel_size)
    fig6.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig6.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

    # Option(K,T) vs. Implied Vol.
    fig7 = fig.add_subplot(826)
    slist = np.linspace(0.01, sigma, points)
    vlist = [function(S0, K, T, t, r, q, sigma) for sigma in slist]
    fig7.plot(slist, vlist)
    fig7.grid()
    fig7.set_title('BSM %s Option Value vs. Volatility' % optiontype, color = color_plots, fontsize = title_size)
    fig7.set_xlabel('Volatility $\sigma$', color = color_plots, fontsize = xlabel_size)
    fig7.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. Volatility vs. Underlying Price
    slist = np.linspace(0.01, sigma, points)
    s0list = np.linspace(S0 - 20, S0 + 20, points)
    V = np.zeros((len(s0list), len(slist)), dtype=np.float)
    for j in range(len(slist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], K, T, t, r, q, slist[j])

    fig8 = fig.add_subplot(828, projection="3d")
    x, y = np.meshgrid(slist, s0list)
    fig8.patch.set_alpha(0.0)
    fig8.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig8.set_title('BSM %s Option Value vs. Volatility vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig8.set_xlabel('Volatility $\sigma$', color = color_plots, fontsize = xlabel_size)
    fig8.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig8.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

"""#### Monte Carlo Option Sensitivity"""

def mc_plot_values(function,S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims, optiontype):
    fig = plt.figure(figsize=(30,90))
    points = 100

    # Option(K,T) vs. Strike
    fig1 = fig.add_subplot(821)
    klist = np.linspace(K-30, K+30, points)
    vlist = [function(S0, K, T, t, r, q, sigma,n_paths,n_steps, n_sims) for K in klist]
    fig1.plot(klist, vlist)
    fig1.grid()
    fig1.set_title('Monte Carlo %s Option Value vs. Strike' % optiontype, color = color_plots, fontsize = title_size)
    fig1.set_xlabel('Strike $K$', color = color_plots, fontsize = xlabel_size)
    fig1.set_ylabel('%s Option Value' % optiontype, color = color_plots, fontsize = ylabel_size)

    # Option(K,T) vs. Strike vs. Underlying Price
    klist = np.linspace(K-20, K+20, points)
    s0list = np.linspace(S0 - 20, S0 + 20, points)
    V = np.zeros((len(s0list), len(klist)), dtype=np.float)
    for j in range(len(klist)):
      for i in range(len(s0list)):
        V[i, j] = function(s0list[i], klist[j], T, t, r, q, sigma,n_paths,n_steps, n_sims)

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
    vlist = [function(S0, K, T, t, r, q, sigma,n_paths,n_steps, n_sims) for T in tlist]
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
        V[i, j] = function(s0list[i], K, tlist[j], t, r, q, sigma,n_paths,n_steps, n_sims)

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
    vlist = [function(S0, K, T, t, r, q, sigma,n_paths,n_steps, n_sims) for r in rlist]
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
        V[i, j] = function(s0list[i], K, T, t, rlist[j], q, sigma,n_paths,n_steps, n_sims)

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
    vlist = [function(S0, K, T, t, r, q, sigma,n_paths,n_steps, n_sims) for sigma in slist]
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
        V[i, j] = function(s0list[i], K, T, t, r, q, slist[j],n_paths,n_steps, n_sims)

    fig8 = fig.add_subplot(828, projection="3d")
    x, y = np.meshgrid(slist, s0list)
    fig8.patch.set_alpha(0.0)
    fig8.plot_wireframe(x, y, V, linewidth=1.0, color = color_plots) #cmap = cm.RdYlGn, for surface plotting
    fig8.set_title('Monte Carlo %s Option Value vs. Volatility vs. Underlying Price' % optiontype, color = color_plots, fontsize = title_size)
    fig8.set_xlabel('Volatility $\sigma$', color = color_plots, fontsize = xlabel_size)
    fig8.set_ylabel('Stock/Underlying Price ($)', color = color_plots, fontsize = ylabel_size)
    fig8.set_zlabel('%s Option Value' % optiontype, color = color_plots, fontsize = zlabel_size)

"""### **Variance Reduction**

#### Conventional Implementation
"""

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

"""#### JAX Implementation"""

# Call Options:
def mc_call_cv(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims):
  c_cv = []
  c_cv_temp = []
  c_upper_cv = []
  c_lower_cv = []
  K = np.full(n_paths,K)
  payoff_0 = np.zeros(n_paths)
  for i in range(1,n_sims):
    S = gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)
    S_arithm = jnp.mean(S, axis=0)
    S_geom = jnp.exp(jnp.mean(jnp.log(S), axis=0))
    payoff_arithm = jnp.mean(jnp.exp(-r*T) * jnp.maximum(S_arithm - K, payoff_0))
    payoff_geom = jnp.mean(jnp.exp(-r*T) * jnp.maximum(S_geom - K, payoff_0))
    c_cv_temp.append(jnp.mean(payoff_arithm - payoff_geom))
    c_cv.append(c_cv_temp)
    c_upper_cv.append(np.mean(c_cv) + 1.96 * np.std(c_cv)/np.sqrt(i))
    c_lower_cv.append(np.mean(c_cv) - 1.96 * np.std(c_cv)/np.sqrt(i))
  return c_cv, c_upper_cv, c_lower_cv

# Put Options:
def mc_put_cv(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims):
  p_cv = []
  p_cv_temp = []
  p_upper_cv = []
  p_lower_cv = []
  K = np.full(n_paths,K)
  payoff_0 = np.zeros(n_paths)
  for i in range(1,n_sims):
    S = gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)
    S_arithm = jnp.mean(S, axis=0)
    S_geom = jnp.exp(jnp.mean(jnp.log(S), axis=0))
    payoff_arithm = jnp.mean(jnp.exp(-r*T) * jnp.maximum(K - S_arithm, payoff_0))
    payoff_geom = jnp.mean(jnp.exp(-r*T) * jnp.maximum(K - S_geom, payoff_0))
    p_cv_temp.append(jnp.mean(payoff_arithm - payoff_geom))
    p_cv.append(p_cv_temp)
    p_upper_cv.append(np.mean(p_cv_temp) + 1.96 * np.std(p_cv_temp)/np.sqrt(i))
    p_lower_cv.append(np.mean(p_cv_temp) - 1.96 * np.std(p_cv_temp)/np.sqrt(i))
  return p_cv, p_upper_cv, p_lower_cv

"""## **Performance Tests**

### Black-Scholes vs. Geometric Monte Carlo

#### Test Cases:
"""

# Call Options
print('- - - - - - - - - - - - - - - - -')
n_paths = 100000  # Number of paths per simulation
n_steps = 252     # Number of steps N (1 monitoring point per business day)
n_sims = 100      # Number of simulations

for i in range(95,110,5):
  bs = bsm_call(S0, i, T, t, r, q, sigma)
  mc = mc_call_geom(S0, i, T, t, r, q, sigma, n_paths, n_steps, n_sims)
  error = np.abs((bs-mc)/bs)*100
  print('G_c(K=%d,T=%d):'%(i,T))
  print('Black-Scholes:',bs)
  print('Monte Carlo:', mc)
  print('Error: %f%%' % error)
  print('- - - - - - - - - - - - - - - - -')

# Put Options
print('- - - - - - - - - - - - - - - - -')
n_paths = 100000  # Number of paths per simulation
n_steps = 504     # Number of steps N (2 monitoring points per day)
n_sims = 100      # Number of simulations

for i in range(95,110,5):
  bs = bsm_put(S0, i, T, t, r, q, sigma)
  mc = mc_put_geom(S0, i, T, t, r, q, sigma, n_paths, n_steps, n_sims)
  error = np.abs((bs-mc)/bs)*100
  print('G_p(K=%d,T=%d):'%(i,T))
  print('Black-Scholes:',repr(bs))
  print('Monte Carlo:',repr(mc))
  print('Error:',repr(error),'%')
  print('- - - - - - - - - - - - - - - - -')

"""#### Final Results:

**Call Options:**
- - - - - - - - - - - - - - - - -
G_c(K=95,T=1):
Black-Scholes: 12.508538481017892
Monte Carlo: 12.447389441700023
Error: 0.488858%
- - - - - - - - - - - - - - - - -
G_c(K=100,T=1):
Black-Scholes: 9.61215966961383
Monte Carlo: 9.638977665500253
Error: 0.279001%
- - - - - - - - - - - - - - - - -
G_c(K=105,T=1):
Black-Scholes: 7.185865725921304
Monte Carlo: 7.171154551742235
Error: 0.204724%
- - - - - - - - - - - - - - - - -
**Put Options:**
- - - - - - - - - - - - - - - - -
G_p(K=95,T=1):
Black-Scholes: 2.194652455717922
Monte Carlo: 2.1827188484716005
Error: 0.543758407634428 %
- - - - - - - - - - - - - - - - -
G_p(K=100,T=1):
Black-Scholes: 3.6018135264391438
Monte Carlo: 3.5923289672198133
Error: 0.26332732524071434 %
- - - - - - - - - - - - - - - - -
G_p(K=105,T=1):
Black-Scholes: 5.479059464871914
Monte Carlo: 5.455180840115154
Error: 0.4358161270169417 %
- - - - - - - - - - - - - - - - -

### Asian call options with Arithmetic Average: Linetsky Test Cases

#### Introduction:

The arithmetic average option pricer is benchmarked against the test cases in Table B of the Linetsky paper 1 *Exotic spectra, Risk magazine, April 2002. V. Linetsky* (reproduced below):

**%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%**

**C &emsp; r &emsp; &emsp; &emsp; σ &emsp; &emsp; T &emsp; S0 &emsp; EE &emsp; &emsp; &emsp; MC &emsp; &emsp; &emsp; &emsp; &emsp; %Err**

1 &emsp; 0.0200 &emsp; 0.10 &emsp; 1 &emsp; 2.0 &emsp; 0.05602 &emsp; 0.0559860415 &emsp; 0.017

2 &emsp; 0.1800 &emsp; 0.30 &emsp; 1 &emsp; 2.0 &emsp; 0.21850 &emsp; 0.2183875466 &emsp; 0.059

3 &emsp; 0.0125 &emsp; 0.25 &emsp; 2 &emsp; 2.0 &emsp; 0.17250 &emsp; 0.1722687410 &emsp; 0.063

4 &emsp; 0.0500 &emsp; 0.50 &emsp; 1 &emsp; 1.9 &emsp; 0.19330 &emsp; 0.1931737903 &emsp; 0.084

5 &emsp; 0.0500 &emsp; 0.50 &emsp; 1 &emsp; 2.0 &emsp; 0.24650 &emsp; 0.2464156905 &emsp; 0.095

6 &emsp; 0.0500 &emsp; 0.50 &emsp; 1 &emsp; 2.1 &emsp; 0.30640 &emsp; 0.3062203648 &emsp; 0.106

7 &emsp; 0.0500 &emsp; 0.50 &emsp; 2 &emsp; 2.0 &emsp; 0.35030 &emsp; 0.3500952190 &emsp; 0.146

**%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%**

**Notes:**
* *(EE = Eigenfunction Expansion i.e. the Black-Scholes analytic result in 
this algorithm; MC = Monte-Carlo estimate);*
* *All test cases have a strike K = 2.0 and a dividend yield q = 0.0;*

#### Test Cases:
"""

print('- - - - - - - - - - - - - - - - -')
# Test Case 1 Parameters:
# r = 0.02
# sigma = 0.1
# T = 1
# S0 = 2.0
n_paths = 100000  # Number of paths per simulation
n_steps = 252     # Number of steps M (fixed for T = 1 year)
n_sims = 1000     # Number of simulations
lntk1 = 0.0559860415

mc = mc_call_arithm(2.0, 2.0, 1, 0.0, 0.02, 0.0, 0.1,n_paths,n_steps,n_sims)
error = np.abs((lntk1-mc)/lntk1)*100
print('Case 1 (r=0.02, sigma=0.10, T=1, S0=2.0):')
print('G_c(K=%d,T=%d):'%(2.0,1))
print('Linetsky:',repr(lntk1))
print('Monte Carlo:',repr(mc))
print('Error:',repr(error),'%')

print('- - - - - - - - - - - - - - - - -')

print('- - - - - - - - - - - - - - - - -')
# Test Case 2 Parameters:
# r = 0.18
# sigma = 0.30
# T = 1.0
# S0 = 2.0
n_paths = 100000  # Number of paths per simulation
n_steps = 252     # Number of steps M (fixed for T = 1 year)
n_sims = 1000     # Number of simulations
lntk2 = 0.2183875466

mc = mc_call_arithm(2.0, 2.0, 1.0, 0.0, 0.18, 0.0, 0.3,n_paths,n_steps,n_sims)
error = np.abs((lntk2-mc)/lntk2)*100
print('Case 2 (r = 0.18, sigma = 0.30, T = 1, S0 = 2.0):')
print('G_c(K=%d,T=%d):'%(2.0,1))
print('Linetsky:',repr(lntk2))
print('Monte Carlo:',repr(mc))
print('Error:',repr(error),'%')

print('- - - - - - - - - - - - - - - - -')

print('- - - - - - - - - - - - - - - - -')
# Test Case 3 Parameters:
# r = 0.0125
# sigma = 0.25
# T = 2.0
# S0 = 2.0
n_paths = 100000  # Number of paths per simulation
n_steps = 504     # Number of steps M (fixed for T = 2 years)
n_sims = 1000     # Number of simulations
lntk3 = 0.1722687410

mc = mc_call_arithm(2.0, 2.0, 2.0, 0.0, 0.0125, 0.0, 0.25,n_paths,n_steps,n_sims)
error = np.abs((lntk3-mc)/lntk3)*100
print('Case 3 (r = 0.0125, sigma = 0.25, T = 2, S0 = 2.0):')
print('G_c(K=%d,T=%d):'%(2.0,2))
print('Linetsky:',repr(lntk3))
print('Monte Carlo:',repr(mc))
print('Error:',repr(error),'%')

print('- - - - - - - - - - - - - - - - -')

print('- - - - - - - - - - - - - - - - -')
# Test Case 4 Parameters:
# r = 0.05
# sigma = 0.50
# T = 1.0
# S0 = 1.9
n_paths = 100000  # Number of paths per simulation
n_steps = 252     # Number of steps M (fixed for T = 1 year)
n_sims = 1000     # Number of simulations
lntk4 = 0.1931737903

mc = mc_call_arithm(1.9, 2.0, 1.0, 0.0, 0.05, 0.0, 0.50,n_paths,n_steps,n_sims)
error = np.abs((lntk4-mc)/lntk4)*100
print('Case 4 (r = 0.05, sigma = 0.50, T = 1, S0 = 1.9):')
print('G_c(K=%d,T=%d):'%(2.0,1))
print('Linetsky:',repr(lntk4))
print('Monte Carlo:',repr(mc))
print('Error:',repr(error),'%')

print('- - - - - - - - - - - - - - - - -')

print('- - - - - - - - - - - - - - - - -')
# Test Case 5 Parameters:
# r = 0.05
# sigma = 0.50
# T = 1.0
# S0 = 2.0
n_paths = 100000  # Number of paths per simulation
n_steps = 252     # Number of steps M (fixed for T = 1 year)
n_sims = 1000     # Number of simulations
lntk5 = 0.2464156905

mc = mc_call_arithm(2.0, 2.0, 1.0, 0.0, 0.05, q, 0.50,n_paths,n_steps,n_sims)
error = np.abs((lntk5-mc)/lntk5)*100
print('Case 5 (r = 0.05, sigma = 0.50, T = 1, S0 = 2.0):')
print('G_c(K=%d,T=%d):'%(2.0,1))
print('Linetsky:',repr(lntk5))
print('Monte Carlo:',repr(mc))
print('Error:',repr(error),'%')

print('- - - - - - - - - - - - - - - - -')

print('- - - - - - - - - - - - - - - - -')
# Test Case 6 Parameters:
# r = 0.05
# sigma = 0.50
# T = 1.0
# S0 = 2.1
n_paths = 100000  # Number of paths per simulation
n_steps = 252     # Number of steps M (fixed for T = 1 year)
n_sims = 1000     # Number of simulations
lntk6 = 0.3062203648

mc = mc_call_arithm(2.1, 2.0, 1.0, 0.0, 0.05, 0.0, 0.50,n_paths,n_steps,n_sims)
error = np.abs((lntk6-mc)/lntk6)*100
print('Case 6 (r = 0.05, sigma = 0.50, T = 1, S0 = 2.1):')
print('G_c(K=%d,T=%d):'%(2.0,1))
print('Linetsky:',repr(lntk6))
print('Monte Carlo:',repr(mc))
print('Error:',repr(error),'%')

print('- - - - - - - - - - - - - - - - -')

print('- - - - - - - - - - - - - - - - -')
# Test Case 7 Parameters:
# r = 0.05
# sigma = 0.50
# T = 2.0
# S0 = 2.0
n_paths = 100000  # Number of paths per simulation
n_steps = 504     # Number of steps M (fixed for T = 1 year)
n_sims = 1000     # Number of simulations
lntk7 = 0.3500952190

mc = mc_call_arithm(2.0, 2.0, 2.0, 0.0, 0.05, q, 0.50,n_paths,n_steps,n_sims)
error = np.abs((lntk7-mc)/lntk7)*100
print('Case 7 (r = 0.05, sigma = 0.50, T = 2.0, S0 = 2.0):')
print('G_c(K=%d,T=%d):'%(2.0,2))
print('Linetsky:',repr(lntk7))
print('Monte Carlo:',repr(mc))
print('Error:',repr(error),'%')

print('- - - - - - - - - - - - - - - - -')

"""#### Final Results:

- - - - - - - - - - - - - - - - -
Case 1 (r=0.02, sigma=0.10, T=1, S0=2.0):
G_c(K=2,T=1):
Linetsky: 0.0559860415
Monte Carlo: 0.0559778385271486
Error: 0.014651817902507759 %
- - - - - - - - - - - - - - - - -
Case 2 (r = 0.18, sigma = 0.30, T = 1, S0 = 2.0):
G_c(K=2,T=1):
Linetsky: 0.2183875466
Monte Carlo: 0.2170470505500451
Error: 0.6138152430505384 %
- - - - - - - - - - - - - - - - -
Case 3 (r = 0.0125, sigma = 0.25, T = 2, S0 = 2.0):
G_c(K=2,T=2):
Linetsky: 0.172268741
Monte Carlo: 0.1726129894916586
Error: 0.19983224446888723 %
- - - - - - - - - - - - - - - - -
Case 4 (r = 0.05, sigma = 0.50, T = 1, S0 = 1.9):
G_c(K=2,T=1):
Linetsky: 0.1931737903
Monte Carlo: 0.1917548655071399
Error: 0.7345327700287397 %
- - - - - - - - - - - - - - - - -
Case 5 (r = 0.05, sigma = 0.50, T = 1, S0 = 2.0):
G_c(K=2,T=1):
Linetsky: 0.2464156905
Monte Carlo: 0.2472542514106814
Error: 0.3403033747485359 %
- - - - - - - - - - - - - - - - -
Case 6 (r = 0.05, sigma = 0.50, T = 1, S0 = 2.1):
G_c(K=2,T=1):
Linetsky: 0.3062203648
Monte Carlo: 0.3066361640492379
Error: 0.1357843230019895 %
- - - - - - - - - - - - - - - - -
Case 7 (r = 0.05, sigma = 0.50, T = 2.0, S0 = 2.0):
G_c(K=2,T=2):
Linetsky: 0.350095219
Monte Carlo: 0.3475691276119277
Error: 0.7215440974280498 %
- - - - - - - - - - - - - - - - -

### Monte Carlo vs. Number of Trials

The results of 1,50000,100000,150000, ... ,1000000 paths are compared to the Black-Scholes theoretical price. The errors are then log-plotted against the number of paths in order to determine the sensitivity of the pricing algorithm to the number of trials.
"""

# Iterations Cases:
it_paths = [1,100,1000,10000,100000,1000000]

# Call Options:
bsm_c = bsm_call(S0, K, T, t, r, q, sigma)
print('The Black-Scholes Theoretical Call Price is:',bsm_c)
errors_c = []

for i in range(1,len(it_cases)):
  mc_c = mc_call_geom(S0, K, T, t, r, q, sigma, it_paths[i], n_steps, n_sims)
  print('For %d trials, the Monte Carlo Call Price estimate is:' %it_paths[i],mc_c)
  error_c = 100 * (np.abs(mc_c[0]-bsm_c)/bsm_c)
  errors_c.append(error_c)

# Errors v. Simulations Log-Plot:
errors_v_runs = plt.figure()
ax = errors_v_runs.add_subplot(111)
ax.set_title("Errors vs. Simulations (Call)")
ax.plot(np.log(it_cases), np.log(errors_c), "o-")

# Put Options:
bsm_p = bsm_put(S0, K, T, t, r, q, sigma)
print('The Black-Scholes Theoretical Call Price is:',bsm_p)
errors_p = []

for i in range(0,len(it_cases)):
  mc_p = mc_put_geom(S0, K, T, t, r, q, sigma, it_cases[i], n_steps, n_sims)
  print('For %d iterations, the Monte Carlo Call Price estimate is:' %it_cases[i],mc_p)
  error_p = 100 * (np.abs(mc_p - bsm_p)/bsm_p)
  errors_p.append(error_p)

# Errors v. Simulations Log-Plot:
errors_v_runs = plt.figure()
ax = errors_v_runs.add_subplot(111)
ax.set_title("Errors vs. Simulations (Call)")
ax.plot(np.log(it_cases), np.log(errors_p), "o-")

"""### Multiple Control Variates"""

# Base parameters:
S0 = 100          # Spot price
K = 95            # Strike price
T = 1.0           # Maturity in years
r = 0.15          # Risk-free rate
q = 0.0           # Option dividend yield
sigma = 0.3       # Volatility
n_paths = 100000       # Number of paths per simulation
n_steps = 252   # Number of steps M
n_sims = 100     # Number of simulations

# Call Options:
mc_call_cv_0 = mc_call_cv(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims)
c_cv_values = mc_call_cv_0[0]
c_cv_upper = mc_call_cv_0[1]
c_cv_lower = mc_call_cv_0[2]
data_call = [c_cv_upper, c_cv_lower, c_cv_values]
for i in range(0,len(data_call)):
  plt.plot(data_call[i])
plt.title('$Arithmetic - Geometric Call Option Value Difference',color=color_plots,fontsize=title_size)
plt.xlabel('$\Delta$t', color=color_plots, fontsize=xlabel_size)
plt.ylabel('Difference', color=color_plots, fontsize=ylabel_size)
plt.show()

# Put Options:
mc_put_cv = mc_put_control_variates(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims)
p_cv_values = mc_put_cv[0]
p_cv_upper = mc_put_cv[1]
p_cvv_lower = mc_put_cv[2]
data_put = [p_cv_upper, p_cvv_lower, p_cv_values]
for i in range(0,len(data_put)):
  plt.plot(data_put[i])
plt.title('$Arithmetic - Geometric Put Option Value Difference',color=color_plots,fontsize=title_size)
plt.xlabel('$\Delta$t', color=color_plots, fontsize=xlabel_size)
plt.ylabel('Difference', color=color_plots, fontsize=ylabel_size)
plt.show()

"""## **Hedging Test**

### Daily Data Generation

The timesteps factor was set to timesteps = 1,000,000.

Assumptions:
*  Each path is refined to approximately 11 updates/second. The calculation assumes that price movements occur evenly throughout a 24-hour cycle.
* After-hours trading does not move the closing price of the security each day i.e. the price of the security at 9:30am (open market) is the same as the closing price the previous day.
* The security pays no dividend.
"""

# Define parameters:
n_steps = 1000000 # Granularity of ~11 movements/sec
n_paths = 5 # each path represents 1 business day, and a full business week of trading is generated for a hypothetical security.
S_geometric = []
call = []
call_deltas = []
call_gammas = []
time_vector = np.arange(1,n_paths+1,1) # For plotting

# Initialize values:
S0 = 100
K = 95
t = 0.0
r = 0.15
q = 0.0
sigma = 0.3
n_paths = 100000  # Number of paths per simulation
n_steps = 252     # Number of steps N (1 monitoring point per business day)
n_sims = 100      # Number of simulations

# Modified Geometric Brownian Path Generator; The last element of each price path becomes the first element of the next:
def gbm_paths_hedging(S0,T,r,q,sigma,seed,n_paths,n_steps):
  np.random.seed(seed)    
  dt = T/n_steps
  bt = np.random.randn(int(n_paths), int(n_steps))
  S = S0 * np.cumprod((np.exp(sigma * np.sqrt(dt) * bt + (r - q - 0.5 * (sigma**2)) * dt)), axis = 1)
  S[0][0] = S0
  for i in range(0,len(S)-1):
    S[i+1][0] = S[i][-1]
  return S

# Plot Price Paths:
for i in range(0,5):
  plt.plot(np.transpose(daily_data[i]), label = ('Day %s' % str(i+1)))
  plt.title('Price Path, Day %s' % str(i+1), color = color_plots, fontsize = title_size)
  plt.xlabel('Timesteps', color = color_plots, fontsize = xlabel_size)
  plt.ylabel("Underlying Price ($)", color = color_plots, fontsize = ylabel_size)
  plt.legend(fontsize = legend_size)
  plt.show()

"""### Portfolio Sensitivity

#### Options on Daily Stock Data
"""

# Calculate option prices:
n_paths = 100000
n_steps = 252
for i in range(0,5):
  S = gbm_paths_hedging(100,1.0,0.15,0.0,0.3,seed,n_paths,n_steps)
  S_geom = np.exp(np.mean(np.log(S)))
  S_geometric.append(S_geom)
  call.append(np.exp(-r * (T-t) * max(S_geom - K, 0)))
  call_delta.append(bsm_delta(,K,T,,r,q,sigma,'Call')

"""#### BSM Values (Control)

**1st Order Greeks: Sensitivity to Strike and Stock/Underlying Price**
"""

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(30, 30))
fig.suptitle('1st Order Greeks: Sensitivity to Strike and Stock/Underlying Price. $S_0 = %d$' % S0, color = color_plots, fontsize=title_size)
#Variables:
klist = [85, 95, 100, 105, 115]
S0list = np.arange(50,150)

plt.subplot(321)
for i in klist:
    c = [bsm_delta(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
    p = [bsm_delta(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
    plt.plot(S0list, c, label = ("Delta Call K=%i" % i ))
    plt.plot(S0list, p, label = ("Delta Put K=%i" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Delta", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(322)
for i in klist:
    c = [bsm_gamma(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
    p = [bsm_gamma(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
    plt.plot(S0list, c, label = ("Gamma Call K=%i" % i ))
    plt.plot(S0list, p, label = ("Gamma Put K=%i" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Gamma", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(323)
for i in klist:
    c = [bsm_vega(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
    p = [bsm_vega(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
    plt.plot(S0list, c, label = ("Vega Call K=%i" % i ))
    plt.plot(S0list, p, label = ("Vega Put K=%i" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Vega", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(324)
for i in klist:
    c = [bsm_rho(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
    p = [bsm_rho(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
    plt.plot(S0list, c, label = ("Rho Call K=%i" % i ))
    plt.plot(S0list, p, label = ("Rho Put K=%i" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Rho", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(325)
for i in klist:
    c = [bsm_theta(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
    p = [bsm_theta(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
    plt.plot(S0list, c, label = ("Theta Call K=%i" % i ))
    plt.plot(S0list, p, label = ("Theta Put K=%i" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Theta", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(326)
for i in klist:
    c = [bsm_charm(S0, i, T, t, r, q, sigma, "Call") for S0 in S0list]
    p = [bsm_charm(S0, i, T, t, r, q, sigma, "Put") for S0 in S0list]
    plt.plot(S0list, c, label = ("Charm Call K=%i" % i ))
    plt.plot(S0list, p, label = ("Charm Put K=%i" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Charm", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)
plt.show()

"""**1st Order Greeks: Sensitivity to Risk-Free Rate + Stock/Underlying Price**"""

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(30, 30))
fig.suptitle('1st Order Greeks: Sensitivity to Risk-Free Rate + Stock/Underlying Price', color = color_plots, fontsize=title_size, fontweight='bold')

# Variables:
rlist = [0.0,0.01,0.1]
S0list = np.arange(50,150)
K = 95
r = 0.15
sigma = 0.3
T = 1.0
t = 0.0
q = 0.0

plt.subplot(321)
for i in rlist:
  c = [bsm_delta(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
  p = [bsm_delta(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
  plt.plot(c, label = ("Delta Call r=%.2f" % i ))
  plt.plot(p, label = ("Delta Put r=%.2f" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Delta", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(322)
for i in rlist:
  c = [bsm_gamma(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
  p = [bsm_gamma(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
  plt.plot(c, label = ("Gamma Call r=%.2f" % i ))
  plt.plot(p, label = ("Gamma Put r=%.2f" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Gamma", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(323)
for i in rlist:
  c = [bsm_vega(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
  p = [bsm_vega(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
  plt.plot(c, label = ("Vega Call r=%.2f" % i ))
  plt.plot(p, label = ("Vega Put r=%.2f" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Vega", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(324)
for i in rlist:
  c = [bsm_rho(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
  p = [bsm_rho(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
  plt.plot(c, label = ("Rho Call r=%.2f" % i ))
  plt.plot(p, label = ("Rho Put r=%.2f" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Rho", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(325)
for i in rlist:
  c = [bsm_theta(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
  p = [bsm_theta(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
  plt.plot(c, label = ("Theta Call r=%.2f" % i ))
  plt.plot(p, label = ("Theta Put r=%.2f" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Theta", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

plt.subplot(326)
for i in rlist:
  c = [bsm_charm(S0, K, T, t, i, q, sigma, "Call") for S0 in S0list]
  p = [bsm_charm(S0, K, T, t, i, q, sigma, "Put") for S0 in S0list]
  plt.plot(c, label = ("Charm Call r=%.2f" % i ))
  plt.plot(p, label = ("Charm Put r=%.2f" % i ))

plt.xlabel('Stock/Underlying Price ($)', color = color_plots, fontsize = xlabel_size)
plt.ylabel("Charm", color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)
plt.show()

"""**BSM Option Sensitivity Definitions Tests**"""

# Variables:
S0 = 100
K = 95
T = 1.0
t = 0.0
r = 0.01
q = 0.0
sigma = 0.1
n_paths = 1000# Number of paths per simulation
n_steps = 252   # Number of steps M
n_sims = 7      # Number of simulations
mc_call_cv(S0,K,T,t,r,q,sigma,n_paths,n_steps,n_sims)

"""**Monte Carlo Option Sensitivity Definitions Tests**"""

# Greeks Plot tool for all Greeks:
S0 = np.linspace(50, 150, 100)
K = np.linspace(95.0,95.0, 100)
T = np.linspace(0.1, 1.0, 100)
t = np.linspace(0.0, 0.0, 100)
r = np.linspace(0.15, 0.15, 100)
q = np.linspace(0.0,0.0,100)
sigma = np.linspace(0.30, 0.30, 100)

x, y  = np.meshgrid(S0, T)
fig = plt.figure(figsize=(30,90))
greeks = [bsm_delta,bsm_gamma,bsm_vega,bsm_theta,bsm_rho,bsm_vanna,bsm_vomma, bsm_phi, bsm_charm]
greeks_names = ['Delta','Gamma','Vega','Theta','Rho','Vanna','Volga','Phi','Charm']
for i in range(len(greeks)):
    ax = fig.add_subplot(len(greeks), 2, i+1, projection='3d')
    ax.patch.set_alpha(0.0)
    greeks_plot_tool(greeks[i],"Stock/Underlying Price ($)", "Expiry (T)", S0, K, T, t, r, q, sigma, x, y, "Call", greeks_names[i], ax)
plt.show()

"""### Hedged-Unhedged Portfolio Variability"""

# Shift values:
def convert(lst): 
    return [ i - 50 for i in lst ]

# Calculate Deltas

for j in range(0,len(daily_data)):
  for i in range(0,n_steps):
    call_delta = bsm_delta(daily_data[j][i],K,T,,r,q,sigma,'Call')
  call_delta = call_delta/n_steps
  call_deltas.append(call_delta)
print(call_deltas)

for j in range(0,len(daily_data)):
  for i in range(0,n_steps):
    call_gamma = bsm_gamma(daily_data[j][i],K,T,,r,q,sigma,'Call')
  call_gamma = call_delta/n_steps
  call_gammas.append(call_gamma)
print(call_gamams)
# Results
#print('Call Deltas:')
#print(call_deltas)

# Hedged Portfolio
hedged = []
for i in range(0,len(daily_data)):
  hedged.append(call[i]-call_deltas[i]*S_geometric[i])

#hedged = convert(hedged)
std_unhedged = np.std(call)
std_hedged = np.std(hedged)
print('The standard deviation of the unhedged options portfolio is:',std_unhedged)
print('The standard deviation of the hedged options portfolio is:',std_hedged)

# Plots
plt.plot(time_vector,call, label = 'Unhedged Portfolio')
plt.plot(time_vector,hedged, label = 'Hedged Portfolio')
plt.title('Variability in a Hedged and Unhedged Options Portfolio', color = color_plots, fontsize = title_size)
plt.xlabel('Option Price', color = color_plots, fontsize = xlabel_size)
plt.ylabel('Time (days)', color = color_plots, fontsize = ylabel_size)
plt.legend(fontsize = legend_size)

"""## **Final Project Deliverables**"""

# Test MC Engine:

S0 = 100          # Spot price
K = 95            # Strike price
T = 1.0           # Maturity in years
r = 0.15          # Risk-free rate
q = 0.0           # Option dividend yield
sigma = 0.3       # Volatility
n_paths = 100000       # Number of paths per simulation
n_steps = 252   # Number of steps M
n_sims = 7     # Number of simulations

# mc_plot_values(mc_call_geom,S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims,'Call')
#for i in range(1,10):
#   print(mc_call_geom(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims))
#   print(bsm_call(S0,K,T,t,r,q,sigma))

# Test MC Engine:

S0 = 100          # Spot price
K = 95            # Strike price
T = 1.0           # Maturity in years
r = 0.15          # Risk-free rate
q = 0.0           # Option dividend yield
sigma = 0.3       # Volatility
n_paths = 100000       # Number of paths per simulation
n_steps = 252   # Number of steps M
n_sims = 100     # Number of simulations

plt.plot(mc_call_cv(S0,K,T,t,r,q,sigma,n_paths,n_steps,n_sims))
