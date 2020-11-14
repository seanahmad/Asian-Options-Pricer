import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

class AsianCall:

    def d1(self, asset_price, strike_price, risk_free_rate, volatility, dt):
        return (1 / ((volatility / np.sqrt(3)) * np.sqrt(dt))) * (np.log((asset_price * np.exp(0.5 * risk_free_rate * dt - (volatility ** 2) * dt / 12)) / strike_price) + 0.5 * ((volatility / np.sqrt(3)) ** 2) * (dt))

    def d2(self, asset_price, strike_price, risk_free_rate, volatility, dt):
        return (1 / ((volatility / np.sqrt(3)) * np.sqrt(dt))) * (np.log((asset_price * np.exp(0.5 * risk_free_rate * dt - (volatility ** 2) * dt / 12)) / strike_price) - 0.5 * ((volatility / np.sqrt(3)) ** 2) * (dt))

    def price(self, asset_price, d1, strike_price, d2, risk_free_rate, volatility, dt):
        n1 = stats.norm.cdf(d1)
        n2 = stats.norm.cdf(d2)
        call_price = np.exp(-(risk_free_rate*dt))*((asset_price * np.exp(0.5 * risk_free_rate * dt - (volatility ** 2) * dt / 12))*n1 - strike_price*n2)
        print(call_price)
        return call_price

    def delta(self, d1, dt):
        return np.exp(-dt) * stats.norm.cdf(d1)

    def exercise_prob(self):
        return 1 - stats.norm.cdf(((self.strike_price - self.asset_price) - (self.drift*self.asset_price*self.dt))/((self.volatility*self.asset_price)*(self.dt**.5)))

    def __init__(self, asset_price, strike_price, volatility, expiration_date, risk_free_rate, drift):
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.expiration_date = expiration_date
        self.risk_free_rate = risk_free_rate
        self.drift = drift
        # Calculate delta t
        dt = np.busday_count(datetime.date.today(), expiration_date) / 252
        # Calculate d1
        d1 = self.d1(asset_price, strike_price, risk_free_rate, volatility, dt)
        # Calculate d2
        d2 = self.d2(asset_price, strike_price, risk_free_rate, volatility, dt)
        self.dt = dt
        self.price = self.price(asset_price, d1, strike_price, d2, risk_free_rate, volatility, dt)
        self.delta = self.delta(d1, dt)

class AsianPut:

    def d1(self, asset_price, strike_price, risk_free_rate, volatility, dt):
        return (1 / ((volatility / np.sqrt(3)) * np.sqrt(dt))) * (np.log((asset_price * np.exp(0.5 * risk_free_rate * dt - (volatility ** 2) * dt / 12)) / strike_price) + 0.5 * ((volatility / np.sqrt(3)) ** 2) * (dt))

    def d2(self, asset_price, strike_price, risk_free_rate, volatility, dt):
        return (1 / ((volatility / np.sqrt(3)) * np.sqrt(dt))) * (np.log((asset_price * np.exp(0.5 * risk_free_rate * dt - (volatility ** 2) * dt / 12)) / strike_price) - 0.5 * ((volatility / np.sqrt(3)) ** 2) * (dt))

    def price(self, asset_price, d1, strike_price, d2, risk_free_rate, volatility, dt):
        n1 = stats.norm.cdf(-d1)
        n2 = stats.norm.cdf(-d2)
        put_price = np.exp(-(risk_free_rate*dt))*(strike_price*n2 - (asset_price * np.exp(0.5 * risk_free_rate * dt - ((volatility ** 2) * dt / 12)))*n1)
        #print(put_price)
        return put_price

    def delta(self, d1):
        return stats.norm.cdf(-d1)

    def __init__(self, asset_price, strike_price, volatility, expiration_date, risk_free_rate, drift):
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.expiration_date = expiration_date
        self.risk_free_rate = risk_free_rate
        self.drift = drift
        dt = np.busday_count(datetime.date.today(), expiration_date) / 252
        d1 = self.d1(asset_price, strike_price, risk_free_rate, volatility, dt)
        d2 = self.d2(asset_price, strike_price, risk_free_rate, volatility, dt)
        self.dt = dt
        self.price = self.price(asset_price, d1, strike_price, d2, risk_free_rate, volatility, dt)
        self.delta = self.delta(d1)
        self.asset_price = asset_price

class LiveOptionsGraph:

    def time_step(self, z):
        dt = np.busday_count(datetime.date.today(), self.expiration_date) / 252
        if dt != 0:
            if self.type == 'call':
                price_step = (self.volatility * np.sqrt(dt) * (np.random.normal(0.0, 1.0) + (self.risk_free_rate - 0.5 * (self.volatility ** 2)) * dt))
                ao = AsianCall(self.asset_prices[self.index] + price_step, self.strike_price, self.volatility, self.expiration_date, self.risk_free_rate, self.drift)
            elif self.type == 'put':
                price_step = (self.volatility * np.sqrt(dt) + (np.random.normal(0.0, 1.0) + (self.risk_free_rate - 0.5 * (self.volatility ** 2)) * dt))
                ao = AsianPut(self.asset_prices[self.index] + price_step, self.strike_price, self.volatility, self.expiration_date, self.risk_free_rate, self.drift)
        self.option_prices.append(ao.price)
        self.deltas.append(ao.delta)
        self.hedged.append(ao.price - ao.delta*(self.asset_prices[self.index]+price_step))
        #print(self.hedged)
        self.index_set.append(self.index)
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        #self.axs[3].cla()
        self.axs[0].plot(self.index_set, self.option_prices, label='Naked Portfolio', c='b')
        self.axs[1].plot(self.index_set, self.deltas, label='$\Delta_C$', c='gray')

        if self.type == 'call':
            if self.strike_price <= self.asset_prices[self.index]:
                self.axs[2].plot(self.index_set, self.asset_prices + price_step, label='$S_t$', c='g')
                self.axs[2].axhline(y=self.strike_price, label='$K_C$', c='gray')
                #self.axs[3].plot(self.index_set, self.asset_prices, label='Asset Price', c='g')
                #self.axs[3].axhline(y=self.strike_price, label='Call Strike', c='gray')

            else:
                self.axs[2].plot(self.index_set, self.asset_prices, label='$S_t$', c='r')
                self.axs[2].axhline(y=self.strike_price, label='$K_C$', c='gray')
        elif self.type == 'put':
            if self.strike_price < self.asset_prices[self.index]:
                self.axs[2].plot(self.index_set, self.asset_prices, label='$S_t$', c='r')
                self.axs[2].axhline(y=self.strike_price, label='$K_P$', c='gray')
            else:
                self.axs[2].plot(self.index_set, self.asset_prices, label='$S_t$', c='g')
                self.axs[2].axhline(y=self.strike_price, label='$K_P$', c='gray')

        self.axs[0].legend(loc='upper left')
        self.axs[1].legend(loc='upper left')
        self.axs[2].legend(loc='upper left')
        self.asset_prices.append(ao.asset_price)
        self.index = self.index + 1
        # Helps display time decay
        self.expiration_date = self.expiration_date - timedelta(days=1)

    def __init__(self, european_option, type):
        self.index = 0
        self.asset_price = european_option.asset_price
        self.strike_price = european_option.strike_price
        self.volatility = european_option.volatility
        self.expiration_date = european_option.expiration_date
        self.risk_free_rate = european_option.risk_free_rate
        self.drift = european_option.drift
        self.type = type
        self.index_set = []
        self.option_prices = []
        self.hedged = []
        self.asset_prices = [european_option.asset_price]
        self.deltas = []
        plt.style.use('default')
        self.fig, self.axs = plt.subplots(3)
        self.ani = FuncAnimation(plt.gcf(), self.time_step, 1)
        plt.tight_layout()
        plt.show()

ac = AsianCall(100, 95, 0.3, datetime.date(2021, 4, 27), 0.15, .2)
#ap = AsianPut(100, 95, 0.3, datetime.date(2021, 4, 27), 0.15, .2)
lgc = LiveOptionsGraph(ac, 'call')
#lgp = LiveOptionsGraph(ap, 'call')