import matplotlib.pyplot as plt
import yfinance as yf
from ta.momentum import RSIIndicator
import numpy as np
import pandas as pd

# Note to self:
# Benefits of parameterising:
# Reuse the same functions across tickers
# Avoid magic numbers which leads to better code quality and experiments

# _ throwaway variable:
# Ignoring the value -> Keeps intent clear

# Function to donwload the data of a stock
# Multi_level_index = False: Single-level column index (keeps things tidy)
def downloadData(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate, auto_adjust=True, multi_level_index=False)
    return data

# Downloaded data includes:
# Close -> Closing price of the day
# Open > First traded price of the day
# High/Low -> Highest and lowest price of the day
# Volume -> Number of shares traded that day
apple = downloadData('AAPL', '2018-01-01', '2025-01-01')
print("Data for Apple stock:")
print(apple)

# Function that builds mean reversion indicators
# maWindow = Moving average window = 20
# bandK = width multiplier for Bollinger bands (default = 2)
# RSI = relative strength index lookback = 14
def indicators(data, maWindow=20, bandK=2, rsiLen=14):
    # Fresh data frame only containing the Close column
    # Keeps things simple and fast for SMA, bands and RSI calculations
    df = data[['Close']].copy()

    # Calculating Simple Moving Average across a window
    df['SMA'] = df['Close'].rolling(window=maWindow).mean()

    # Calculating the rolling standard deviation across a window
    std = df['Close'].rolling(window=maWindow).std().replace(0, np.nan)
    df['STD'] = std

    # Calculating Bollinger Bands
    # The bands are roughly a 95% envelope if given normal returns  (with bandK=2)
    df['UpperBand'] = df['SMA'] + (bandK * std)
    df['LowerBand'] = df['SMA'] - (bandK * std)

    # Calculating the z-score
    # Represents the number of standard deviations the close is from the SMA
    # z = -2 -> near/through LB -> Oversold
    # z = +2 -> near/through UP -> Overbought
    df['z'] = (df['Close'] - df['SMA']) / std

    # Calculating the relative strength index across a window
    df['rsi'] = RSIIndicator(df['Close'], window=rsiLen).rsi()

    # Drop the rows where the indicator is NaN (usually the warm-up period in the beginning)
    return df.dropna()

improvedData = indicators(apple)
print("Data with indictors:")
print(improvedData)

# Function that plots the relative strength index of the data
def plotRSI(data, startDate=None, endDate=None):

    # Date slice is chosen if both dates are given
    if startDate and endDate:
        plotData = data.loc[startDate:endDate]
    else:
        plotData = data

    # Plotting the RSI against time
    plt.figure(figsize=(12, 6))
    plt.plot(plotData['rsi'], label='RSI', color='blue')
    plt.ylim(0, 100)

    # Plotting RSI bounds
    plt.axhline(70, ls='--')
    plt.axhline(30, ls='--')

    plt.title('RSI')
    plt.legend()
    plt.grid()
    plt.show()

plotRSI(improvedData)

# "z-score + RSI" that outputs either entry or exit
# It enters when it's very cheap and exits when the price reverts toward the mean
def makeSignals(df, zEntry=-2.0, zExit=-0.5, useRSI=True):

    # zEntry = cheap-ness criteria to enter
    # zExit = how close back to the mean is required to exit
    # useRSI = optionally require RSI oversold

    # Prevent mutation of caller
    signal = df.copy()

    # Entry condition: is today's z=score at/under the entry threshold?
    longEntry = (signal['z'] <= zEntry) & (signal['z'].shift(1) > zEntry)

    # Optional filter: only enter when RSI is oversold too
    #  Tightens entries: fewer trades and higher selectivity
    if useRSI:
        longEntry &= (signal['rsi'] < 30)

    # Exit condition: cross up through the zExit of today and ystd
    # To take profit near the mean
    # Change condition here for fewer restraints
    longExit = (signal['z'] >= zExit) & (signal['z'].shift(1) < zExit)

    # Shift one bar forward to prevent look ahead
    # Fill warm-up NaNs with False
    # Output is a boolean column
    signal['entry'] = longEntry.shift(1).fillna(False)
    signal['exit'] = longExit.shift(1).fillna(False)

    # Return the data frame with entry and exit columns
    return signal

# Helper function that turns event signals (entry, exit) into position series (0, 1)
def positionFromSignals(sig: pd.DataFrame):

    # "Target change" array for this bar
    # Exit = True -> 0
    # Entry = True -> 1
    # Else -> NaN
    # Exit has priority over entry if both are True
    change = np.where(sig['exit'], 0, np.where(sig['entry'], 1, np.nan))

    # Make a series with the same date index
    # ffill() carries the last non-NaN target forward
    # fillna(0) sets any leading NaNs to 0 (the NaNs are usually in front)
    pos = pd.Series(change, index=sig.index).ffill().fillna(0)

    # Cast to integer
    return pos.astype(int)

'''
Note to self:
Review this calculations and understand it further.
'''

# Function that computes the equity curve and its metrics
# An equity curve is a graph that visualises the performance of an investment or trading account
# over time.
# x-axis = time; y-axis = account's value
def equityCurveAndMetrics(sig: pd.DataFrame, costBPS: float = 1.0, rfAnnual: float = 0.0, periods: int = 252):

    # Builds 0/1 position series from the entry/exit booleans
    pos = positionFromSignals(sig)

    # Raw returns are daily close-to-close returns with the first day set to 0
    # pct_change stands for percentage change
    rawRet = sig['Close'].pct_change().fillna(0.0)

    # Apply ystd's position to today's return to avoid lookahead
    # Flat before the first signal
    # pos = end of the day position (0 = flat, 1 = long)
    # * rawRet = only earn today's market return if you were in the trade as of yesterday's close
    stratGross = pos.shift(1).fillna(0) * rawRet

    # 1 on entry or exit days
    # Else is 0
    # 0, 1 for positions
    turnover = pos.diff().abs().fillna(0)

    # Per-side trading costs: bps -> decimals
    # Charged on each entry/exit day
    costs = turnover * (costBPS / 10000.0)

    # Net daily strategy returns after costs
    dailyRet = stratGross - costs

    # Equity curve starting at 1.o with compounding daily returns
    equity = (1.0 + dailyRet).cumprod()

    # Equity curve's metrics
    n = len(dailyRet) # Sample size
    annFactor = np.sqrt(periods) # Annualisation factor
    rfDaily = rfAnnual / periods # Daily risk-free

    # Sharpe uses population std
    meanExcess = (dailyRet - rfDaily).mean()
    vol = dailyRet.std(ddof=0)
    sharpe = (meanExcess / vol * annFactor) if vol > 0 else np.nan

    # Max drawdown as a positive fraction
    # Standard peak-to-trough
    rollMax = equity.cummax()
    drawdown = equity / rollMax - 1.0
    maxDD = -drawdown.min()

    # CAGR from start equity over 1.o+ years
    # Edge case 0 division is protected
    years = n / periods if periods > 0 else np.nan
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years and years > 0 else np.nan

    # Average time in the market
    exposure = pos.mean()

    # Key statistics
    metrics = {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxDD,
        "AnnVol": vol * annFactor,
        "Exposure": exposure,
        "Trades": int((pos.diff().gt(0)).sum()) # Counts entry events after shift
    }

    # Equity curve, daily returns and metrics dictionary
    return equity, dailyRet, metrics

# Function to plot Equity Curve
# Shows th account value over time
# Start at 1.00 (initial capital) and compound daily returns
def plotEquityCurve(equity: pd.Series, title: str = "Strategy Equity Curve"):
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Equity')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

# Function to plot Drawdown
# Measures how far the equity falls from its prior peak
# Exist to quantify pain and path-risk (not average risk)
# EG:
# Equity: 100 → 120 → 108 → 114 → 130 → 124
# Peaks: 100 → 120 → 120 → 120 → 130 → 130
# DD%: 0% → 0% → −10% → −5% → 0% → −4.6%
def plotDrawdown(equity: pd.Series, title: str = "Drawdown"):
    dd = equity / equity.cummax() - 1.0
    plt.figure(figsize=(12, 3.5))
    plt.plot(dd, label='Drawdown')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

signals = makeSignals(improvedData)
print(signals)

equity, dailyRet, metrics = equityCurveAndMetrics(signals, costBPS=1, rfAnnual=0.0, periods=252)

plotEquityCurve(equity)
plotDrawdown(equity)

# CAGR -> Compound Annual Growth Rate (long-run growth rate)
# Sharpe -> Risk-adjusted return
# MaxDD -> Maximum drawdown (worst peak-to-trough drop in the equity curve)
# AnnVol -> Annualised volatility ("bumpiness" of returns per year)
# Exposure -> Fraction of days in a position
# Trades -> Count of entry events
print(
    f"CAGR: {metrics['CAGR']:.2%} | Sharpe: {metrics['Sharpe']:.2f} | "
    f"MaxDD: {metrics['MaxDD']:.2%} | AnnVol: {metrics['AnnVol']:.2%} | "
    f"Exposure: {metrics['Exposure']:.2%} | Trades: {metrics['Trades']}"
)

# Plotting the Results with Buy and Sell signals
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

    buySignals = plotData[plotData['entry']]
    sellSignals = plotData[plotData['exit']]

    plt.scatter(buySignals.index, buySignals['Close'], color='green', marker='^', s=100, label='Entry', zorder=3)
    plt.scatter(sellSignals.index, sellSignals['Close'], color='red', marker='v', s=100, label='Exit', zorder=3)

    plt.legend()
    plt.title('Mean Reversion (Bollinger + RSI) with Entries/Exits')

    plt.grid()
    plt.show()

plotResultsSignals(signals)

# Function to backtest the algorithm for the long-only strategy
# costBPS = per-side cost in basis points
def backTestLongOnly(sig, costBPS=1):

    # Create a copy
    df = sig.copy()

    # Poitions: 0 = flat and 1 = long
    position = 0
    buyPX = None
    trades = []

    # Loop through rows
    # t = date index
    # row = row
    for t, row in df.iterrows():

        # If flat and entry flag is set, buy at that bar
        # Change the position into long
        # Log the trade
        # Print the process
        if position == 0 and row['entry']:
            buyPX = row['Close']
            position = 1
            trades.append({'type': 'buy', 'date': t, 'price': buyPX})
            print(f"Buying at {buyPX:.2f} on {t.date()}") # Figure out what .2f means

        # If long and exit flag is set,sell at that bar
        # Compute gross return
        # Subtract a round-trip cost
        # Log net return in percent
        # Print the process
        # Change the position to flat
        elif position == 1 and row['exit']:
            sellPX = row['Close']
            gross = sellPX / buyPX - 1
            net = gross - 2 * costBPS / 10000
            trades.append({'type': 'sell', 'date': t, 'price': sellPX, 'return%': net * 100})
            print(f"Selling at {sellPX:.2f} on {t.date()} | Return: {net*100:.2f}%")
            position = 0
            buyPX = None

    # If the trade is still open at the end, force close at the last close
    # And record the returns
    if position == 1:
        t = df.index[-1]
        sellPX = df['Close'].iloc[-1]
        gross = sellPX / buyPX - 1
        net = gross - 2 * costBPS / 10000
        trades.append({'type': 'sell', 'date': t, 'price': sellPX, 'return%': net * 100})
        print(f"Closing open long at end: {sellPX:.2f} on {t.date()} | Return: {net*100:.2f}%")

    # Collect per-trade net% returns (only for sells)
    rets = [x['return%'] for x in trades if x['type'] == 'sell']

    # If trades are completed, print the count
    # Print the average return per trade
    # Print the win rate
    if rets:
        avg = np.mean(rets)
        win = np.mean([r > 0 for r in rets])
        print(f"\nBacktest Results:")
        print(f"Trades: {len(rets)} | Avg Return: {avg:.2f}% | Win Rate: {win:.2%}")

    # Return the trade log
    # A list of dictionaries with buy/sell
    return trades

tradeHistory = backTestLongOnly(signals, costBPS=1)
print(f"Entries: {signals['entry'].sum()} | Exits: {signals['exit'].sum()}")

# Helper function that builds buy-and-hold equity returns from a price series
def buyAndHoldEquity(close: pd.Series):

    # Computes daily percentage return
    # Set first day to 0.0 (it has no returns)
    ret = close.pct_change().fillna(0.0)

    # Start at 1.0 and compound returns to get the equity curve
    eq = (1.0 + ret).cumprod()

    # Return the equity curve series and daily return series
    return eq, ret

# Function to plot the equity curve against the buy-and-hold built
# Buy and hold = buy the asset and don't trade
# The above is the baseline to beat
# Strategy cruve ccompared to this to see if outperforming is possible
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

# Function which is a metrics helper
def metricsFromReturns(dailyRet: pd.Series, rfAnnual: float = 0.0, periods: int = 252):

    # Convert annual risk-free to per-period daily rate
    rfDaily = rfAnnual / periods
    # Annualisation factor
    annFactor = np.sqrt(periods)
    # Number of return observations
    n = len(dailyRet)
    # Equity curve from returns
    eq = (1.0 + dailyRet.fillna(0.0)).cumprod()
    # Sample length in years
    years = n/periods if periods>0 else np.nan
    # CAGR from start to last equity over years
    cagr = (eq.iloc[-1] ** (1/years) - 1) if years and years>0 else np.nan
    # Daily volatility
    vol = dailyRet.std(ddof=0)
    # Sharpe: mean excess daily return / daily vol (annualised)
    sharpe = ((dailyRet - rfDaily).mean() / vol*annFactor) if vol>0 else np.nan
    # Drawdown series below running peak
    dd = eq/eq.cummax() - 1.0
    # Maximum drawdown as a positive fraction
    maxdd = -dd.min()

    # Returning a dictionary of headline stats
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd, "AnnVol": vol*annFactor}

bhEq, bhRet = buyAndHoldEquity(improvedData['Close'])
bhMetrics = metricsFromReturns(bhRet)

plotEquityWithBenchmark(equity, improvedData['Close'])
print("Strategy:", {k: (f"{v:.2%}" if 'CAGR' in k or 'MaxDD' in k or 'AnnVol' in k else f"{v:.2f}") for k,v in metrics.items() if k in ['CAGR','Sharpe','MaxDD','AnnVol']})
print("Buy&Hold:", {k: (f"{v:.2%}" if 'CAGR' in k or 'MaxDD' in k or 'AnnVol' in k else f"{v:.2f}") for k,v in bhMetrics.items()})

# Function which is a transaction-cost sensitivity test
def costSweep(base: pd.DataFrame, zEntry=-2.0, zExit=-0.5, useRSI=True, costGrid=(0, 1, 2, 5, 10)):

    # Results accumulator
    rows = []

    # Build entry/exit signals
    sig = makeSignals(base, zEntry=zEntry, zExit=zExit, useRSI=useRSI)


    # Loop over each assummed cost level
    for c in costGrid:

        # Run the backtest with costBPS = c
        eq, dr, m = equityCurveAndMetrics(sig, costBPS=c)

        # Store the metrics with the cost level
        rows.append({"costBPS": c, **m})

    # Tabulate the row
    df = pd.DataFrame(rows)

    # Print the key columns for a comparison
    print(df[['costBPS', 'CAGR', 'Sharpe', 'MaxDD', 'AnnVol']])

    # Return the full Data Frame
    return df

_ = costSweep(improvedData, zEntry=-2.0, zExit=-0.5, useRSI=True, costGrid=(0, 1, 2, 5, 10))

# Function to search through the data
def gridSearch(rawPrices: pd.DataFrame,
               maWindows=(15, 20, 30),
               zEntries=(-1.5, -2.0, -2.5),
               zExits=(-0.25, -0.5, -0.75),
               useRSI=(True, False),
               costBPS=1):

    # Result accumulator
    rows = []

    # Loop through the moving avergae windows to obtain the indicators
    for w in maWindows:
        ind = indicators(rawPrices, maWindow=w)

        # Nested loops over the z entry, z exit and RSI flag
        for ze in zEntries:
            for zx in zExits:
                for ur in useRSI:

                    # Build entry/exit conditions for this combination
                    sig = makeSignals(ind, zEntry=ze, zExit=zx, useRSI=ur)

                    # Backtest the given combination at the given cost
                    eq, dr, m = equityCurveAndMetrics(sig, costBPS=costBPS)

                    # Record the input parameters and results
                    rows.append({
                        "maWindow": w, "zEntry": ze, "zExit": zx, "useRSI": ur, **m
                    })

    # Tabulate and rank based on Sharpe and then CAGR
    res = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False])

    # Display top 10 results
    print(res.head(10)[["maWindow", "zEntry", "zExit", "useRSI", "Sharpe", "CAGR", "MaxDD", "Exposure"]])

    # Return the complete results
    return res

gs = gridSearch(apple)

# Function that runs strategy parameters across a list of given tickers over a specified date
def batchRun(tickers, start="2018-01-01", end="2025-01-01", maWindow=20, zEntry=-2.0, zExit=-0.5, useRSI=True, costBPS=1):

    # Results accumulator
    out = []

    # Loop through each ticker
    for tk in tickers:

        # Downaload data for each ticker
        px = yf.download(tk, start=start, end=end, auto_adjust=True, multi_level_index=False)
        # Skip emties
        if px.empty:
            continue
        # Build indicators for each ticker
        ind = indicators(px, maWindow=maWindow)
        # Build boolean signals (entry/exit) for each ticker
        sig = makeSignals(ind, zEntry=zEntry, zExit=zExit, useRSI=useRSI)
        # Get metrics
        _, _, m = equityCurveAndMetrics(sig, costBPS=costBPS)
        # Append metrics alongside ticker
        out.append({"ticker": tk, **m})

    # Build a table and sort according to Sharpe
    df = pd.DataFrame(out).sort_values("Sharpe", ascending=False)

    # Print out stats per ticker
    print(df[["ticker", "Sharpe", "MaxDD", "Exposure"]])
    # Check performance
    print("\nMedian Sharpe:", df["Sharpe"].median())

    # Return the full results Data Frame
    return df

_ = batchRun(["AAPL","MSFT","AMZN","GOOGL","META","NVDA","KO","PEP","JPM","XOM"])

'''
Not Used Reference Code:
# Function to plot the stock with the indicators
def plotResults(data, startDate=None, endDate=None, window=20):

    # If the start and end dates are given, slice the data frame within those dates.
    # Otherwise, use the whole dataset for plotting
    if startDate and endDate:
        plotData = data.loc[startDate:endDate]
    else:
        plotData = data

    # Set the figure size as 12x6 inches
    plt.figure(figsize=(12, 6))

    # Plots price vs date
    plt.plot(plotData['Close'], label='Price', color='blue')
    plt.plot(plotData['SMA'], label=f'SMA {window} days', color='green')
    plt.plot(plotData['UpperBand'], label='UpperBand', color='red', linestyle='--')
    plt.plot(plotData['LowerBand'], label='LowerBand', color='red', linestyle='--')

    plt.title('Mean Reversion Indicators')
    plt.legend()
    plt.grid()

    # Renders the figure
    plt.show()

plotResults(improvedData)

# Not used in this algorithm
# "Classic band touch + RSI" strategy using a single column
# It is a "flip to the other extreme" exit
# This would result in either holding the long too long or never exiting
def signalResults(data):
    df = data.copy()

    # Event creation:
    # Creates a +1 signal where RSI<30 and price is below LB
    # Otherwise, it is NaN
    df['signal'] = np.where(
        ((df['rsi'] < 30) &
         (df['Close'] < df['LowerBand']))
        , 1, np.nan)

    # Overwrites signal with -1 where RSI>70 and price is above UP
    # Otherise, keep the bar with the value as it is
    # If both conditions are true (rarely), -1 is the output
    df['signal'] = np.where(
        ((df['rsi'] > 70) &
         (df['Close'] > df['UpperBand']))
        , -1, df['signal'])

    # Shifts all signals by 1 bar to avoid look-ahead
    # Enables the algorithm to act in the next bar
    df['signal'] = df['signal'].shift()

    # Reaplces all the NaN with 0
    df['signal'] = df['signal'].fillna(0)

    return df

# Not used anymore
def backtestDataframe(data):
    position = 0
    percentageChange = []
    trades = []
    buyPrice = 0

    for i in data.index:
        close = data.loc[i, 'Close']

        if data.loc[i, 'signal'] == 1:
            if (position == 0):
                buyPrice = close
                position = 1
                trades.append({'type': 'buy', 'date': i, 'price': buyPrice})
                print(f"Buying at {buyPrice:.2f} on {i.date()}")

        elif data.loc[i, 'signal'] == -1:
            if (position == 1):
                sellPrice = close
                position = 0

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
'''