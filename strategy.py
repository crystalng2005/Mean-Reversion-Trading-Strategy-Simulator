import matplotlib.pyplot as plt
import yfinance as yf
from ta.momentum import RSIIndicator
import numpy as np

# Function to donwload the data of a
def downloadData(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate, multi_level_index=False)
    return data

apple = downloadData('MSFT', '2018-01-01', '2025-01-01')
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
        plotData = data.copy()

    plt.figure(figsize=(12, 6))
    plt.plot(plotData['Close'], label='Closing Price', color='blue', alpha=0.7)
    plt.plot(plotData['SMA'], label=f'SMA {window} days', color='green', alpha=0.7)
    plt.plot(plotData['UpperBand'], label='Upper Band', color='red', linestyle='--', alpha=0.7)
    plt.plot(plotData['LowerBand'], label='Lower Band', color='red', linestyle='--', alpha=0.7)

    buySignals = plotData[plotData['signal'] == 1]
    sellSignals = plotData[plotData['signal'] == -1]

    plt.scatter(buySignals.index, buySignals['Close'], color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sellSignals.index, sellSignals['Close'], color='red', marker='v', s=100, label='Sell Signal')

    plt.legend()
    plt.title('Mean Reversion Indicators with Trading Signals')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    plt.grid()
    plt.show()

plotResultsSignals(signalResults(improvedData))

def backtestDataframe(data):
    position = 0
    percentageChange = []
    trades = []
    buyPrice = 0

    for i in data.index:
        # close = data['Close'][i]
        # date = data['Date'][i]
        close = data.loc[i, 'Close']

        if data.loc[i, 'signal'] == 1:
            if (position == 0):
                buyPrice = close
                position = 1
                # data.at[i, 'buyDate'] = date
                trades.append({'type': 'buy', 'date': i, 'price': buyPrice})
                print(f"Buying at {buyPrice:.2f} on {i.date()}")

        elif data.loc[i, 'signal'] == -1:
            if (position == 1):
                sellPrice = close
                position = 0
                # data.at[i, 'sellDate'] = date


                pc = (sellPrice/buyPrice-1)*100
                percentageChange.append(pc)
                trades.append({'type': 'sell', 'date': i, 'price': sellPrice, 'return%': pc})
                print(f"Selling at {sellPrice:.2f} on {i.date()} | Return: {pc:.2f}%")

    if len(percentageChange) > 0:
        avgReturn = sum(percentageChange)/len(percentageChange)
        winRate = len([x for x in percentageChange if x>0])/len(percentageChange)
        print(f"\nBacktest Results:")
        print(f"Total Trades: {len(percentageChange)}")
        print(f"Average Return: {avgReturn:.2f}%")
        print(f"Winning Rate: {winRate:.2%}")

    return trades

signals = signalResults(improvedData)
tradeHistory = backtestDataframe(signals)

print(f"Buy signals: {sum(signals['signal'] == 1)}")
print(f"Sell signals: {sum(signals['signal'] == -1)}")









