import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Step 1: Fetch Financial Data with Error Handling
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL"]
raw_data = yf.download(tickers, start="2020-01-01", end="2024-01-01")

# Step 2: Print Raw Data Structure for Debugging
print("Raw data structure:\n", raw_data.head())

# Extract 'Close' Prices Instead of 'Adj Close'
try:
    data = raw_data['Close']  # FIXED: Corrected from ['Price']['Close'] to ['Close']
except KeyError:
    raise ValueError("Error: 'Close' prices not found. Check the downloaded data structure.")

# Step 3: Handle Missing Data - Drop Empty Columns
if data.empty:
    raise ValueError("No valid stock data retrieved. Ensure tickers are correct and data is available.")

data.dropna(axis=1, how='all', inplace=True)
valid_tickers = list(data.columns)
print(f"Using valid tickers: {valid_tickers}")

# Step 4: Print Processed Data for Verification
print("\nProcessed Data (Close Prices):\n", data.head())

# Step 5: Calculate Returns & Risk
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Step 6: Monte Carlo Simulation for Portfolio Optimization
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_list = []

for i in range(num_portfolios):
    weights = np.random.random(len(valid_tickers))
    weights /= np.sum(weights)
    weights_list.append(weights)

    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = results[0, i] / results[1, i]  # Sharpe Ratio

# Find the best portfolio (max Sharpe Ratio)
max_sharpe_idx = np.argmax(results[2])
best_return, best_risk, best_sharpe = results[:, max_sharpe_idx]
best_weights = weights_list[max_sharpe_idx]

# Step 7: Optimization Using Scipy
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(valid_tickers)))
initial_guess = [1./len(valid_tickers)] * len(valid_tickers)

optimal_weights = minimize(portfolio_volatility, initial_guess, args=(mean_returns, cov_matrix),
                           method='SLSQP', bounds=bounds, constraints=constraints)

# Step 8: Visualizing Results
plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap="viridis", marker="o")
plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Return")
plt.colorbar(label="Sharpe Ratio")
plt.scatter(best_risk, best_return, color="red", marker="*", s=200, label="Optimal Portfolio")
plt.legend()
plt.show()

# Display optimized portfolio weights
optimized_df = pd.DataFrame({
    "Stock": valid_tickers,
    "Optimized Weight": optimal_weights.x
})

print("\nOptimized Portfolio Weights:\n", optimized_df)
