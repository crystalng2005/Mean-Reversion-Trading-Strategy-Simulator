import matplotlib.pyplot as plt
import yfinance as yf
from ta.momentum import RSIIndicator
import numpy as np

# Function to donwload the data of a
def downloadData(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate, multi_level_index=False)
    return data

apple = downloadData('AAPL', '2018-01-01', '2025-01-01')
print(apple)

def indicators(data):
    window = 20

    df = data[['Close']].copy()

    df['SMA'] = df['Close'].rolling(window=window).mean()

    # Bollinger Bands
    df['STD'] = df['Close'].rolling(window=window).std()
    df['UpperBand'] = df['SMA'] + (df['STD'] * 2)
    df['LowerBand'] = df['SMA'] - (df['STD'] * 2)

    df['rsi'] = RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    # There is another way to manually calculate the RSI

    return df

improvedData = indicators(apple)
print(improvedData)

def plotResults(data, startDate=None, endDate=None, window=20):
    if startDate and endDate:
        plotData = data.loc[startDate:endDate]
    else:
        plotData = data

    plt.figure(figsize=(12, 6))
    plt.plot(plotData['Close'], label='Price', color='blue')
    plt.plot(plotData['SMA'], label=f'SMA {window} days', color='green')
    plt.plot(plotData['UpperBand'], label='UpperBand', color='red', linestyle='--')
    plt.plot(plotData['LowerBand'], label='LowerBand', color='red', linestyle='--')
    # plt.plot(plotData['STD'], label='STD', color='red', linestyle='--')

    plt.title('Mean Reversion Indicators')
    plt.legend()
    plt.grid()
    plt.show()



def plotRSI(data, startDate=None, endDate=None, window=20):
    if startDate and endDate:
        plotData = data.loc[startDate:endDate]
    else:
        plotData = data

    plt.figure(figsize=(12, 6))
    plt.plot(plotData['rsi'], label='RSI', color='blue')

    plt.title('RSI')
    plt.legend()
    plt.grid()
    plt.show()

plotResults(improvedData)
plotRSI(improvedData)

def signalResults(data):
    df = data.copy()

    df['signal'] = np.where(
        ((df['rsi'] < 30) &
         (df['Close'] < df['LowerBand']))
        , 1, np.nan)

    df['signal'] = np.where(
        ((df['rsi'] > 70) &
         (df['Close'] > df['UpperBand']))
        , -1, df['signal'])

    df['signal'] = df['signal'].shift()
    df['signal'] = df['signal'].fillna(0)

    return df

# print(improvedData['rsi'] < 30)
# print(improvedData)
# print(improvedData['Close'] < improvedData['LowerBand'])
print(signalResults(improvedData))

def plotResultsSignals(data, startDate=None, endDate=None, window=20):
    if startDate and endDate:
        plotData = data.loc[startDate:endDate]
    else:
        plotData = data

    plt.figure(figsize=(12, 6))
    plt.plot(plotData['Close'], label='Closing Price', color='blue')
    plt.plot(plotData['SMA'], label=f'SMA {window} days', color='green')
    plt.plot(plotData['UpperBand'], label='Upper Band', color='red', linestyle='--')
    plt.plot(plotData['LowerBand'], label='Lower Band', color='red', linestyle='--')

    buySignals = data[data['signal'] == 1]
    sellSignals = data[data['signal'] == -1]

    plt.scatter(buySignals.index, buySignals['Close'], color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sellSignals.index, sellSignals['Close'], color='green', marker='v', s=100, label='Sell Signal')

    plt.legend()
    plt.title('Mean Reversion Indicators with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid()
    plt.show()

plotResultsSignals(signalResults(improvedData))







