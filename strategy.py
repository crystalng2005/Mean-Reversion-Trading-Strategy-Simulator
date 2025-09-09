import matplotlib.pyplot as plt
import yfinance as yf
from ta.momentum import RSIIndicator
import numpy as np
import pandas as pd

# Function to donwload the data of a
def downloadData(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate, multi_level_index=False)
    return data

apple = downloadData('MSFT', '2018-01-01', '2025-01-01')
print(apple)

def indicators(data, maWindow=20, bandK=2, rsiLen=14):
    df = data[['Close']].copy()
    df['SMA'] = df['Close'].rolling(window=maWindow).mean()
    std = df['Close'].rolling(window=maWindow).std().replace(0, np.nan)
    df['STD'] = std
    df['UpperBand'] = df['SMA'] + (bandK * std)
    df['LowerBand'] = df['SMA'] - (bandK * std)
    df['z'] = (df['Close'] - df['SMA']) / std
    df['rsi'] = RSIIndicator(df['Close'].squeeze(), window=rsiLen).rsi()
    return df.dropna()

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

# Not used anymore
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

def makeSignals(df, zEntry=-2.0, zExit=-0.5, useRSI=True):
    signal = df.copy()

    longEntry = (signal['z'] <= zEntry)

    if useRSI:
        longEntry &= (signal['rsi'] < 30)

    longExit = (signal['z'] >= zExit) | (signal['Close'] >= signal['SMA'])

    signal['entry'] = longEntry.shift(1).fillna(False)
    signal['exit'] = longExit.shift(1).fillna(False)
    return signal

def positionFromSignals(sig: pd.DataFrame):
    change = np.where(sig['exit'], 0, np.where(sig['entry'], 1, np.nan))
    pos = pd.Series(change, index=sig.index).ffill().fillna(0)
    return pos.astype(int)

def equityCurveAndMetrics(sig: pd.DataFrame, costBPS: float = 1.0, rfAnnual: float = 0.0, periods: int = 252):
    # Figure out why the above has : and stuff afterwards -- what is the purpopse of this
    # Python formatting? I'm curious
    pos = positionFromSignals(sig)

    rawRet = sig['Close'].pct_change().fillna(0.0)

    stratGross = pos.shift(1).fillna(0) * rawRet

    turnover = pos.diff().abs().fillna(0)
    costs = turnover * (costBPS / 10000.0)

    dailyRet = stratGross - costs

    equity = (1.0 + dailyRet).cumprod()

    n = len(dailyRet)
    annFactor = np.sqrt(periods)
    rfDaily = rfAnnual / periods

    meanExcess = (dailyRet - rfDaily).mean()
    vol = dailyRet.std(ddof=0)
    sharpe = (meanExcess / vol * annFactor) if vol > 0 else np.nan

    rollMax = equity.cummax()
    drawdown = equity / rollMax - 1.0
    maxDD = -drawdown.min()

    years = n / periods if periods > 0 else np.nan
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years and years > 0 else np.nan

    exposure = pos.mean()

    metrics = {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxDD,
        "AnnVol": vol * annFactor,
        "Exposure": exposure,
        "Trades": int(sig['entry'].sum())
    }

    return equity, dailyRet, metrics

def plotEquityCurve(equity: pd.Series, title: str = "Strategy Equity Curve"):
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Equity')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

def plotDrawdown(equity: pd.Series, title: str = "Drawdown"):
    dd = equity / equity.cummax() - 1.0
    plt.figure(figsize=(12, 3.5))
    plt.plot(dd, label='Drawdown')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

# print(signalResults(improvedData))
signals = makeSignals(improvedData)
print(signals)

equity, dailyRet, metrics = equityCurveAndMetrics(signals, costBPS=1, rfAnnual=0.0, periods=252)

plotEquityCurve(equity)
plotDrawdown(equity)

print(
    f"CAGR: {metrics['CAGR']:.2%} | Sharpe: {metrics['Sharpe']:.2f} | "
    f"MaxDD: {metrics['MaxDD']:.2%} | AnnVol: {metrics['AnnVol']:.2%} | "
    f"Exposure: {metrics['Exposure']:.2%} | Trades: {metrics['Trades']}"
)


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

    # buySignals = plotData[plotData['signal'] == 1]
    # sellSignals = plotData[plotData['signal'] == -1]

    buySignals = plotData[plotData['entry']]
    sellSignals = plotData[plotData['exit']]

    plt.scatter(buySignals.index, buySignals['Close'], color='green', marker='^', s=100, label='Entry', zorder=3)
    plt.scatter(sellSignals.index, sellSignals['Close'], color='red', marker='v', s=100, label='Exit', zorder=3)

    plt.legend()
    plt.title('Mean Reversion (Bollinger + RSI) with Entries/Exits')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    plt.grid()
    plt.show()

plotResultsSignals(signals)

# Not used anymore
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



# signals = signalResults(improvedData)
# tradeHistory = backtestDataframe(signals)
#
# print(f"Buy signals: {sum(signals['signal'] == 1)}")
# print(f"Sell signals: {sum(signals['signal'] == -1)}")


def backTestLongOnly(sig, costBPS=1):
    df = sig.copy()
    position = 0
    buyPX = None
    trades = []

    for t, row in df.iterrows():
        if position == 0 and row['entry']:
            buyPX = row['Close']
            position = 1
            trades.append({'type': 'buy', 'date': t, 'price': buyPX})
            print(f"Buying at {buyPX:.2f} on {t.date()}") # Figure out what .2f means

        elif position == 1 and row['exit']:
            sellPX = row['Close']
            gross = sellPX / buyPX - 1
            net = gross - 2 * costBPS / 10000
            trades.append({'type': 'sell', 'date': t, 'price': sellPX, 'return%': net * 100})
            print(f"Selling at {sellPX:.2f} on {t.date()} | Return: {net*100:.2f}%")
            position = 0
            buyPX = None

    if position == 1:
        t = df.index[-1]
        sellPX = df['Close'].iloc[-1]
        gross = sellPX / buyPX - 1
        net = gross - 2 * costBPS / 10000
        trades.append({'type': 'sell', 'date': t, 'price': sellPX, 'return%': net * 100})
        print(f"Closing open long at end: {sellPX:.2f} on {t.date()} | Return: {net*100:.2f}%")

    rets = [x['return%'] for x in trades if x['type'] == 'sell']
    if rets:
        avg = np.mean(rets)
        win = np.mean([r > 0 for r in rets])
        print(f"\nBacktest Results:")
        print(f"Trades: {len(rets)} | Avg Return: {avg:.2f}% | Win Rate: {win:.2%}")
    return trades

tradeHistory = backTestLongOnly(signals, costBPS=1)

print(f"Entries: {signals['entry'].sum()} | Exits: {signals['exit'].sum()}")

'''
Concrete “next commits” (copy this plan into your TODO)

Add buy-and-hold benchmark and print both metrics side-by-side.

Add cost_sweep() → dataframe of Sharpe/CAGR vs cost_bps.

Implement grid_search() over (window, zEntry, zExit, useRSI) with a heatmap.

Chronological train/valid/test split + walk-forward option.

Batch runner over a ticker list; aggregate median metrics.

Add pytest with 4 basic tests + GitHub Actions.

Wrap into a CLI; push a README with plots and “How to run”.
'''

def buyAndHoldEquity(close: pd.Series):
    ret = close.pct_change().fillna(0.0)
    eq = (1.0 + ret).cumprod()
    return eq, ret

def plotEquityWithBenchmark(strategyEQ: pd.Series, close: pd.Series, title="Equity vs Buy & Hold"):
    bhEQ, _ = buyAndHoldEquity(close)

    s = strategyEQ / strategyEQ.iloc[0]
    b = bhEQ / bhEQ.iloc[0]
    plt.figure(figsize=(12, 6))
    plt.plot(s, label="Strategy")
    plt.plot(b, label="Buy & Hold")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

def metricsFromReturns(dailyRet: pd.Series, rfAnnual: float = 0.0, periods: int = 252):
    rfDaily = rfAnnual / periods
    annFactor = np.sqrt(periods)
    n = len(dailyRet)
    eq = (1.0 + dailyRet.fillna(0.0)).cumprod()
    years = n/periods if periods>0 else np.nan
    cagr = (eq.iloc[-1] ** (1/years) - 1) if years and years>0 else np.nan
    vol = dailyRet.std(ddof=0)
    sharpe = ((dailyRet - rfDaily).mean() / vol*annFactor) if vol>0 else np.nan
    dd = eq/eq.cummax() - 1.0
    maxdd = -dd.min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd, "AnnVol": vol*annFactor}

bhEq, bhRet = buyAndHoldEquity(improvedData['Close'])
bhMetrics = metricsFromReturns(bhRet)

plotEquityWithBenchmark(equity, improvedData['Close'])
print("Strategy:", {k: (f"{v:.2%}" if 'CAGR' in k or 'MaxDD' in k or 'AnnVol' in k else f"{v:.2f}") for k,v in metrics.items() if k in ['CAGR','Sharpe','MaxDD','AnnVol']})
print("Buy&Hold:", {k: (f"{v:.2%}" if 'CAGR' in k or 'MaxDD' in k or 'AnnVol' in k else f"{v:.2f}") for k,v in bhMetrics.items()})
# Figure out what are these 2 print statements because they look so complicated

def costSweep(improvedDF: pd.DataFrame, zEntry=-2.0, zExit=-0.5, useRSI=True, costGrid=(0, 1, 2, 5, 10)):
    rows = []
    base = improvedDF
    for c in costGrid:
        sig = makeSignals(base, zEntry=zEntry, zExit=zExit, useRSI=useRSI)
        eq, dr, m = equityCurveAndMetrics(sig, costBPS=c)
        rows.append({"costBPS": c, **m})
    df = pd.DataFrame(rows)
    print(df[['costBPS', 'CAGR', 'Sharpe', 'MaxDD', 'AnnVol']])
    return df

_ = costSweep(improvedData, zEntry=-2.0, zExit=-0.5, useRSI=True, costGrid=(0, 1, 2, 5, 10))

def gridSearch(rawPrices: pd.DataFrame,
               maWindows=(15, 20, 30),
               zEntries=(-1.5, -2.0, -2.5),
               zExits=(-0.25, -0.5, -0.75),
               useRSI=(True, False),
               costBPS=1):
    rows = []
    for w in maWindows:
        ind = indicators(rawPrices, maWindow=w)
        for ze in zEntries:
            for zx in zExits:
                for ur in useRSI:
                    sig = makeSignals(ind, zEntry=ze, zExit=zx, useRSI=ur)
                    eq, dr, m = equityCurveAndMetrics(sig, costBPS=costBPS)
                    rows.append({
                        "maWindow": w, "zEntry": ze, "zExit": zx, "useRSI": ur, **m
                    })
    res = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    print(res.head(10)[["maWindow", "zEntry", "zExit", "useRSI", "Sharpe", "CAGR", "MaxDD", "Exposure"]])
    return res

gs = gridSearch(apple)

def batchRun(tickers, start="2018-01-01", end="2025-01-01", maWindow=20, zEntry=-2.0, zExit=-0.5, useRSI=True, costBPS=1):
    out = []
    for tk in tickers:
        px = yf.download(tk, start=start, end=end, multi_level_index=False)
        ind = indicators(px, maWindow=maWindow)
        sig = makeSignals(ind, zEntry=zEntry, zExit=zExit, useRSI=useRSI)
        _, _, m = equityCurveAndMetrics(sig, costBPS=costBPS)
        # Figure out the purpose of _, I mean why?
        out.append({"ticker": tk, **m})
    df = pd.DataFrame(out).sort_values("Sharpe", ascending=False)
    print(df[["ticker", "Sharpe", "MaxDD", "Exposure"]])
    print("\nMedian Sharpe:", df["Sharpe"].median())
    return df

_ = batchRun(["AAPL","MSFT","AMZN","GOOGL","META","NVDA","KO","PEP","JPM","XOM"])