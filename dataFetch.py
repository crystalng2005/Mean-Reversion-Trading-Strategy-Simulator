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