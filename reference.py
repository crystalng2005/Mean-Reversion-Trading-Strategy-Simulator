'''
RESOURCE: https://www.investopedia.com/terms/m/meanreversion.asp
Mean Reversion Definition:
A financial theory that states that asset prices will tend to revert to
their historical mean or average over time

Mean Reversion Strategies:

1. Statistical Analysis
- Z-scores to measure asset price deviation from its mean
- Z-score above 1.5 or below -1.5

2. Pairs Trading
- Identify two correlated assets

3. Volatility
- Context of volatility
- Buying options when volatility is high (expecting it
return to its mean)

4. Risk Management
- Stop-loss orders
- Take-profit points
- Set around mean to manage potential losses and gains

5. Algorithmic Trading
- Mathematical models to predict price movements

Considerations
1. Time horizon
2. Market conditions

Short-term traders -> Intraday data
Long-term traders -> Yearly data

Mean conversion
Effective: Range-bound markets
less: Trending markets

Mean Reversion Formulas
Time frame = investor/trader's time horizon
Mean = sum of price of prices / Number of Observations
Deviation = Price - Mean
(Calculated for each price point)
Standard deviation = SquareRoot(Sum of Square Deviations)/(Number of Observations - 1)
Z-score = Deviation / Standard Deviation
(Measures how many standard deviations an element is from the mean)
Z-score threshold = 1.5 to 2
(over/under valued)

Mean Reversion (used in technical analysis)
- Helps to identify overbought/oversold conditions -> entry/exit points

1. Moving Averages
- Identify mean price over a specified period
- Overbought - price is above the moving average and over a certain threshold
- Oversold - Opposite of above

2. Bollinger Bands
- Middle band (simple moving average)
- Two outer bands (calculated using the standard deviation)
- Prices expected to revert to the middle band

3. Relative Strength Index (RSI)
- 0 to 100
- >70 overbought conditions
- <30 oversold conditions
- Both implying a potential mean conversion

4. Stochastic Oscillator
- Indicator compares a security's closing price to its price range over
a specified period (usually 14 days)
- >80 overbought conditions
- <20 oversold conditions

5. Moving Average Convergence Divergence (MACD)
- Identify changes in the strength, direction, momentum and duration of a trend
- MACD crosses above/below its signal -> sign that the asset is deviating from its mean

Applications
Day Trading and Mean Reversion
- Buying and selling financial instruments within the same trading day

Swing Trading and Mean Reversion
- Positions are held for several days to weeks

Forex Trading  Using Mean Reversion
- Capitalise on currency pairs reverting to their historical mean or average price

Currency Correlations
- Historically correlated currency pairs

Factors for Mean Reversion
- Trader/investor's objectives
- Risk tolerance
- Asset being traded

Trades suited for Mean Reversion Strategies
- Stocks
- Forex
- Commodities
- Exchange-traded funds (ETFs)
- Fixed income instruments

'''
# INITIAL RESEARCH

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import seaborn as sns
import statsmodels as sm
import backtesting
# import FinRL

plt.style.use('fivethirtyeight')

# Doenloading Historical Stock Data
apple = 'AAPL'
microsoft = 'MSFT'
tesla = 'TSLA'
Amazon = 'AMZN'
data = yf.download(apple, start='2020-01-01', end='2023-01-01')
print(data.info())
print(data.head())

# Fetching Real-Time Stock Prices
stock = yf.Ticker(apple) # Creating a ticker object (why what does this do?)
price = stock.history(period="1d")['Close'][0] # Historical data for last trading day (What is considered as a last trading day)
print(f"Real-time price for {apple}: {price}") # Real-time price

# Retrieving Financial Statements
balanceSheet = stock.balance_sheet
incomeStatement = stock.financials
print("Balance Sheet:")
print(balanceSheet.head())
print("Income Statement:")
print(incomeStatement.head())

# Accessing Ticker Information
info = stock.info
print(f"Company: {info['longName']}")
print(f"Sector: {info['sector']}")
print(f"Industry: {info['industry']}")
print(f"Market Cap: {info['marketCap']}")
print(f"P/E Ratio: {info['trailingPE']}")

# Extra notes:
# Algorithm doesn't work in black swan cases and trending markets
# Can fix this using a stop loss but this would impair the effectiveness

# ATTEMPT 1
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

# ATTEMPT 2
# yfinance for stock data
import yfinance as yf
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import datetime

# download historical data for stock via yfinance
# stock name/start date/end date(XXXX-XX-XX)
stocks = 'AAPL'
data = yf.download(stocks, start='2023-01-01', end='2023-12-31')

# This show the first few rows of data FYI
# print(data.head())

# Visualization with matplotlib
plt.figure(figsize=(69, 42))
plt.plot(data['Close'], label='Closing Price')
plt.title(f"{stocks} Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()

# Machine Learning with sklearn

# ATTEMPT 3
# import yfinance as yf
# import pandas_ta as ta
# from backtesting import Backtest, Strategy
#
# data = yf.download("AAPL", start="2020-01-01", end="2025-06-30")
# data['SMA200'] = ta.sma(data['Close'], length=200)
# data['RS10'] = ta.rsi(data['Close'], length=10)
# data.dropna(inplace=True)
#
# class MeanRev(Strategy):
#     def __init__(self):
#         #something
#
#     def next(self):
#         if self.data.RSI10[-1] and self.data.Close[-1] > self.data.SMA200[-1]:
#             self.buy()
#         elif self.data.RSI10[-1] > 70 and self.data.Close[-1] < self.data.SMA[-1]:
#             self.position.close()
#
# bt = Backtest(data, MeanRev, cash=10000)
# stats = bt.run()
# print(stats)
# bt.plot()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def downloadData(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate)
    return data

data = downloadData('AAPL', '2020-01-01', '2023-06-30')
print(data.head())

def calculateIndicators(data, window=20):
    df = data.copy()

    # Simple Moving Average
    # Calculates the mean close price across the window
    df['SMA'] = df['Close'].rolling(window=window).mean()

    # Bollinger Bands
    # Calculating the standard deviation across the window
    df['RollingStd'] = df['Close'].rolling(window=window).std()

    # df['RollingStd'] = df['RollingStd'].replace([0, np.nan], 1e-10)

    df['UpperBand'] = df['SMA'] + (df['RollingStd'] * 2)
    df['LowerBand'] = df['SMA'] - (df['RollingStd'] * 2)

    # Z-Score
    # Indicates the number of standard deviations away from the mean
    # df['Z-Score'] = (df['Close'] - df['SMA']) / df['RollingStd']

    # df.fillna(method='ffill', inplace=True)

    return df

dataWithIndicators = calculateIndicators(data)
print(dataWithIndicators.tail())

def plotMeanReversion(data, startDate=None, endDate=None, window=20):
    if startDate and endDate:
        plotData = data.loc[startDate:endDate]
    else:
        plotData = data

    plt.figure(figsize=(12,6))
    plt.plot(plotData['Close'], label='Price', color='blue')
    plt.plot(plotData['SMA'], label=f'SMA {window} days', color='green')
    plt.plot(plotData['UpperBand'], label='Upper Band', color='red', linestyle='--')
    plt.plot(plotData['LowerBand'], label='Lower Band', color='red', linestyle='--')

    plt.title('Mean Reversion Indicators')
    plt.legend()
    plt.grid()
    plt.show()

plotMeanReversion(dataWithIndicators)

# Backtesting is a method of evaluating a strategy by applying it to historical data to see how
# it would have perfromed in the past.
# To assess  strategy's potential effectiveness and identify areas for improvement pre live trading

# class MeanReversionStrategy(Strategy):
#     window = 20
#     zEntryThreshold = 2.0
#     zExitThreshold = 0.5
#
#     def init(self):
#         close = self.data.Close
#         self.sma = self.I(SMA, close, self.window)
#         self.rollingStd = self.I(lambda x: x.rolling(self.window).std(), close)
#         self.ZScore = (close - self.sma) / self.rollingStd
#
#     def next(self):
#         currentZ = self.ZScore[-1]
#
#         if not self.position:
#             if currentZ < -self.zEntryThreshold:
#                 self.buy()
#
#             elif currentZ > self.zEntryThreshold:
#                 self.sell()
#
#         else:
#             if self.position.isLong and currentZ >= -self.zExitThreshold:
#                 self.position.close()
#
#             elif self.position.isShort and currentZ <= self.zEntryThreshold:
#                 self.position.close()
#
#     def SMA(series, window):
#         return series.rolling(window).mean()
#
# def runBacktest(data, strategy, cash=10000, commission=.002):
#     bt = Backtest(data, strategy, cash=cash, commission=commission)
#     results = bt.run()
#     return results, bt
#
# results, bt = runBacktest(data, MeanReversionStrategy)
# print(results)


# def analyze_results(results, bt):
#     """
#     Analyze and visualize backtest results
#     """
#     print(results)
#
#     # Plot equity curve
#     bt.plot()
#
#     # Plot trades
#     plt.figure(figsize=(12, 6))
#     plt.plot(results._equity_curve.Equity)
#     plt.title('Equity Curve')
#     plt.xlabel('Trade #')
#     plt.ylabel('Equity')
#     plt.grid()
#     plt.show()
#
#     # Print key metrics
#     print(f"Return: {results['Return [%]']}%")
#     print(f"Sharpe Ratio: {results['Sharpe Ratio']}")
#     print(f"Max Drawdown: {results['Max. Drawdown [%]']}%")
#     print(f"Win Rate: {results['Win Rate [%]']}%")
#
#
# analyze_results(results, bt)

# ATTEMPT 4
# Round 2 of figuring every thing out

import yfinance as yf
import matplotlib.pyplot as plt
import math
from ta.momentum import RSIIndicator
import numpy as np

data = yf.download('AAPL', start='2022-01-01', end='2023-01-01').copy()
data = data[['Close']]
print(data.head())

window = 20
data['mean'] = data['Close'].rolling(window).mean()
data['std'] = data['Close'].rolling(window).std()

# Bollinger Bands
data['upperBound'] = data['mean'] + (2*data['std'])
data['lowerBound'] = data['mean'] - (2*data['std'])

# RSI
data['rsi'] = RSIIndicator(data['Close'].squeeze(), window=14).rsi()

data.dropna(inplace=True)
data = data.reset_index(drop=True)

# conditions = [
#     (data['rsi'].values < 30) & (data['Close'].values < data['lowerBound'].values),
#     (data['rsi'].values > 70) & (data['Close'].values > data['upperBound'].values)
# ]
#
# # Create signals - THIS IS THE FIXED LINE
# data['signal'] = np.select(
#     condlist=conditions,  # List of boolean arrays
#     choicelist=['Buy', 'Sell'],  # Corresponding choices
#     default='Hold'  # Default value
# )
#
# data['signal'] = np.where((data['rsi'] < 30) & (data['close'] < data['lowerBound']), 1, np.nan)
#
# data['signal'] = np.where((data['rsi'] > 70) & (data['close'] < data['lowerBound']), -1, data['signal'])

# print(type(data['mean']), type(data['std']))
# data['zScore'] = (data['Close'] - data['mean']) #/ data['std']

print(data.head(20))
print(data.tail())

# df = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
# df['ma_20'] = df.Close.rolling(20).mean()
# df['vol'] = df.Close.rolling(window).std()
# df['upper_bb'] = df.ma_20 + (2*df.vol)
# df['lower_bb'] = df.ma_20 - (2*df.vol)
#
# df['rsi'] = ta.momentum.rsi(df.Close, window=6)
#
# print(df.head())
# print(df.tail())

# import yfinance as yf
#
#
# df = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
#
# window = 20  # define the window size
# df['ma_20'] = df.Close.rolling(window).mean()
# df['vol'] = df.Close.rolling(window).std()
# df['upper_bb'] = df.ma_20 + (2*df.vol)
# df['lower_bb'] = df.ma_20 - (2*df.vol)
#
# # Calculate RSI - method depends on which TA library you're using
# df['rsi'] = RSIIndicator(df['Close'].squeeze(), window=6).rsi()
# # or: df['rsi'] = ta.momentum.rsi(df.Close, window=6)  # if using a different library
#
# print(df.head(20))  # Show first 20 rows to see where indicators start
# print(df.tail())
