"""
Source:https://algotrading101.com/learn/backtesting-py-guide/#:~:text=To%20code%20a%20mean%2Dreversion%20strategy%20with%20Backtesting.py%2C,out%20and%20run%20the%20backtest.

Backtesting.py is an open-source backtesting Python library that allows users to test their trading strategies via code.
Link: https://github.com/kernc/backtesting.py

Why should I use Backtesting.py?

-Backtesting.py is easy to use.
-Backtesting.py is open-sourced.
-Is compatible with forex, crypto, stocks, futures, and more.
-Offers interactive charts.
-Allows for vectorized or event-based backtesting.
-Has a built-in optimizer.
-Is actively maintained.

Why shouldn’t I use Backtesting.py?

-Backtesting.py could use more features.
-Doesn’t support the use of multiple assets at the same time.
-Data that the strategy needs is constrained (OHLCV).
-Is heavily indicator based.
-Complex strategies either can’t work or require hacking to work.
-The documentation could be better.
-The charting should be more customizable.
-Can be easily replaced by its alternatives.

To code a mean-reversion strategy with Backtesting.py, we will first need to obtain the data of the asset we plan to trade. 
Then, we will lay out our strategy logic to make all the steps clear. After that, we will code it out and run the backtest.

The goal of this strategy will be:
-to sell/short the asset if it is trading more than 3 standard deviations above the rolling mean
-to buy/long the asset if it is trading more than 3 standard deviations below the rolling mean.

To perform backtesting with Backtesting.py, you will need to import the Backtest module and pass it the data, 
the strategy class, set initial cash, and the trade commission value.
"""
import yfinance as yf
import pandas as pd # I added this -Crystal

# obtain the data for the HE asset.

he = yf.download("HE", start="2023-01-15", interval="15m")[
    ["Open", "High", "Low", "Close", "Volume"]
]
he.head()

# set up the trading strategy and the initialization logic
# Though the bulit-in SMA indicator was used it could be replaced with other more complicated ones
from backtesting.test import SMA

def std_3(arr, n):
    return pd.Series(arr).rolling(n).std() * 3

class MeanReversion(Strategy):
    roll = 50

    def init(self):
        self.he = self.data.Close

        self.he_mean = self.I(SMA, self.he, self.roll)
        self.he_std = self.I(std_3, self.he, self.roll)
        self.he_upper = self.he_mean + self.he_std
        self.he_lower = self.he_mean - self.he_std

        self.he_close = self.I(SMA, self.he, 1)

# trading logic.
def next(self):

        if self.he_close < self.he_lower:
            self.buy(
                tp = self.he_mean,
            )

        if self.he_close > self.he_upper:
            self.sell(
                tp = self.he_mean,
            )

# Now backtest:
from backtesting import Backtest

bt = Backtest(he, MeanReversion, cash=10000, commission=0.002)
stats = bt.run()
bt.plot()
stats