import yfinance as yf

# download data
# stock name start date and end date(XXXX-XX-XX)
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')