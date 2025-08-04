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

