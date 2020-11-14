import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import stats

def d1(asset_price, strike_price, risk_free_rate, volatility, dt):
    return (1 / ((volatility / np.sqrt(3)) * np.sqrt(dt))) * (np.log(
        (asset_price * np.exp(0.5 * risk_free_rate * dt - (volatility ** 2) * dt / 12)) / strike_price) + 0.5 * (
                                                                      (volatility / np.sqrt(3)) ** 2) * (dt))

def d2(asset_price, strike_price, risk_free_rate, volatility, dt):
    return (1 / ((volatility / np.sqrt(3)) * np.sqrt(dt))) * (np.log(
        (asset_price * np.exp(0.5 * risk_free_rate * dt - (volatility ** 2) * dt / 12)) / strike_price) - 0.5 * (
                                                                      (volatility / np.sqrt(3)) ** 2) * (dt))

def paths(asset_price, risk_free_rate, dividend_yield, volatility, dt, n_paths):
    seed1 = np.random.randint(1)
    n_steps = int(dt * 252)
    s = jnp.exp((risk_free_rate - dividend_yield - volatility ** 2 / 2) * dt + np.sqrt(
        dt) * volatility * jax.random.normal(jax.random.PRNGKey(seed1), (n_steps, n_paths)))
    s = jnp.vstack([np.ones(n_paths), s])
    s = asset_price * jnp.cumprod(s, axis=0)
    return s

def price(asset_price, d1, strike_price, d2, expiration_date, risk_free_rate, dividend_yield, volatility, dt,
          n_paths, n_sims):
    for i in range(1, n_sims):
        seed = np.random.randint(42)
        n_steps = int(dt * 252)
        s = jnp.exp((risk_free_rate - dividend_yield - volatility ** 2 / 2) * dt + np.sqrt(
            dt) * volatility * jax.random.normal(jax.random.PRNGKey(seed), (n_steps, n_paths)))
        s = jnp.vstack([np.ones(n_paths), s])
        s = asset_price * jnp.cumprod(s, axis=0)
        S_geom = jnp.exp(jnp.mean(jnp.log(s), axis=0))
        self.c.append(jnp.mean(
            jnp.exp(-risk_free_rate * expiration_date) * jnp.maximum(S_geom - np.full(n_paths, strike_price),
                                                                     np.zeros(n_paths))))
    call_price = np.mean(self.c)
    return call_price

def delta(d1, dt):
    return np.exp(-dt) * stats.norm.cdf(d1)

def time_step(z):
    dt = self.expiration_date / 252
    if dt != 0:
        if self.type == 'Call':
            ao = AsianCall(self.price_path[self.index], self.strike_price, self.volatility,
                           self.expiration_date, self.risk_free_rate, self.dividend_yield, self.drift, self.n_paths,
                           self.n_sims)
        elif self.type == 'Put':
            ao = AsianPut(self.price_path[self.index], self.strike_price, self.volatility,
                          self.expiration_date, self.risk_free_rate, self.dividend_yield, self.drift, self.n_paths,
                          self.n_sims)
        self.option_prices.append(ao.price)
        self.deltas.append(ao.delta)
        self.asset_prices.append(price_path[self.index])
        self.hedged.append(ao.price - ao.delta * (self.price_path[self.index]))
        self.index_set.append(self.index)
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        self.axs[3].cla()
        self.axs[0].plot(self.index_set, self.option_prices, label='Naked Portfolio', c='b')
        self.axs[1].plot(self.index_set, self.deltas, label='$\Delta_C$', c='gray')
        self.axs[3].plot(self.index_set, self.hedged, label='Hedged Portfolio', c='b')

        # Plot the asset price and strike price on the 3rd plot, green if in the money red if out of the money
        if self.type == 'Call':
            if self.strike_price <= self.price_path[self.index]:
                self.axs[2].plot(self.index_set, self.asset_prices, animated=True, label='$S_t$', c='g')
                self.axs[2].axhline(y=self.strike_price, label='$K_C$', c='gray')
            else:
                self.axs[2].plot(self.index_set, self.asset_prices, animated=True, label='$S_t$', c='r')
                self.axs[2].axhline(y=self.strike_price, label='$K_C$', c='gray')
        elif self.type == 'Put':
            if self.strike_price < self.asset_prices[self.index]:
                self.axs[2].plot(self.index_set, self.asset_prices, animated=True, label='$S_t$', c='r')
                self.axs[2].axhline(y=self.strike_price, label='$K_P$', c='gray')
            else:
                self.axs[2].plot(self.index_set, self.asset_prices, animated=True, label='$S_t$', c='g')
                self.axs[2].axhline(y=self.strike_price, label='$K_P$', c='gray')
                self.axs[0].legend(loc='upper left', color='black', fontsize=10)
                self.axs[1].legend(loc='upper left', color='black', fontsize=10)
                self.axs[2].legend(loc='upper left', color='black', fontsize=10)
                self.axs[3].legend(loc='upper left', color='black', fontsize=10)
                self.index = self.index + 1

                # Helps display time decay
                self.expiration_date = self.expiration_date - dt

'''self.index = 0
plt.style.use('default')
self.fig, self.axs = plt.subplots(4)
self.ani = FuncAnimation(plt.gcf(), self.time_step, 1000)
plt.tight_layout()
plt.show()
'''
data = paths(100,0.15,0.0,0.3,1/252,1000)
print(data)
'''
price_path = AsianCall.paths(AsianCall, 100, 0.15, 0.0, 0.3, 1 / 252, 1)
initial_ac = AsianCall(100, 95, 0.3, 1.0, 0.15, 0.0, 0.2, 1, 7)
# initial_ap = AsianPut(100, 95, 0.3, datetime.date(2021, 4, 29), 0.15, 0.0, 0.2)
lgc = LiveOptionsGraph(initial_ac, 'Call')
# lgp = LiveOptionsGraph(initial_ap, 'Put')
'''