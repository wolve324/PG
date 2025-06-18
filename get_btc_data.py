import yfinance as yf

# Fetch historical data for Bitcoin
btc = yf.download('BTC-USD', start='2015-01-01', end='2025-01-01')

# Save to CSV
btc.to_csv('BTC-USD.csv')
print("✅ BTC historical data saved as BTC-USD.csv")
