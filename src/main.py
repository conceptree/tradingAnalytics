import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# Step 3: Data Collection
# Downloading P&G stock price data
data = yf.download('PG', start='2019-01-01', end='2023-09-30')

# Step 4: Exploratory Data Analysis (EDA)
# Calculating moving averages
short_window = 40
long_window = 100
data['MA_short'] = data['Close'].rolling(window=short_window).mean()
data['MA_long'] = data['Close'].rolling(window=long_window).mean()

# Plotting the closing price with moving averages
plt.figure(figsize=(14,7))
plt.plot(data['Close'], label='Closing Price')
plt.plot(data['MA_short'], label=f'{short_window}-Day MA')
plt.plot(data['MA_long'], label=f'{long_window}-Day MA')
plt.title('P&G Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 5: Backtesting Strategies
# Initializing the signal and position columns
data['Signal'] = 0.0
data['Signal'][short_window:] = np.where(data['MA_short'][short_window:] > data['MA_long'][short_window:], 1.0, 0.0)

# Taking the difference of the signals to generate actual trading orders
data['Positions'] = data['Signal'].diff()

# Initializing the portfolio with $100,000 and an empty DataFrame for holdings
initial_capital = float(100000.0)
positions = pd.DataFrame(index=data.index).fillna(0.0)
positions['PG'] = 100 * data['Signal']   # This assumes buying 100 shares

# Initializing the portfolio with the value owned
portfolio = positions.multiply(data['Close'], axis=0)

# Store the difference in shares owned
pos_diff = positions.diff()

# Adding `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)

# Adding `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()

# Adding `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Adding `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Plotting the total portfolio value over time
plt.figure(figsize=(14,7))
plt.plot(portfolio['total'], label='Portfolio value')
plt.title('Total Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Total Portfolio Value (USD)')
plt.legend()
plt.show()

# Calculating the Sharpe ratio
returns = portfolio['returns']
sharpe_ratio = sqrt(252) * (returns.mean() / returns.std())

print(f"Sharpe Ratio: {sharpe_ratio}")

# Save the portfolio for further use
portfolio_data_path = '/mnt/data/pg_portfolio.csv'
portfolio.to_csv(portfolio_data_path)
print(f"Portfolio data saved to {portfolio_data_path}")

# Returning the path where the portfolio data is saved
portfolio_data_path
