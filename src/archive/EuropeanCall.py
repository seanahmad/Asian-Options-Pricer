import math
import datetime
import numpy as np
from scipy import stats

class EuropeanCall:
    def d1(self, asset_price, strike_price, risk_free_rate, volatility, dt):
        return (math.log((asset_price / strike_price)) + (risk_free_rate + math.pow(volatility, 2) / 2) * dt) / (
                volatility * math.sqrt(dt))

    def d2(self, d1, volatility, dt):
        return d1 - (volatility * math.sqrt(dt))

    def price(self, asset_price, d1, strike_price, d2, risk_free_rate, dt):
        # Calculate NormalCDF for d1 & d2
        n1 = stats.norm.cdf(d1)
        n2 = stats.norm.cdf(d2)
        # Calculate call option price
        return asset_price * n1 - strike_price * (math.exp(-(risk_free_rate * dt))) * n2

    def delta(self, d1):
        return stats.norm.cdf(d1)

    def exercise_prob(self):
        return 1 - stats.norm.cdf(((self.strike_price - self.asset_price) - (self.drift * self.asset_price * self.dt)) / (
                (self.volatility * self.asset_price) * (self.dt ** .5)))

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
        d2 = self.d2(d1, volatility, dt)
        self.dt = dt
        self.price = self.price(asset_price, d1, strike_price, d2, risk_free_rate, dt)
        self.delta = self.delta(d1)
