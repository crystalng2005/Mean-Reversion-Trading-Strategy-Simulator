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
