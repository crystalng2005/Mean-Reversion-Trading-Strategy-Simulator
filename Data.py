# yfinance for stock data
import yfinance as yf
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import datetime

# download historical data for stock
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
