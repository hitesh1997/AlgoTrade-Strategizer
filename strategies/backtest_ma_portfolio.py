import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(data, stock_name):
    """
    Filter data for the specified stock name.
    """
    stock_data = data[data['stock_name'] == stock_name].copy()
    return stock_data

def calculate_moving_averages(stock_data):
    """
    Calculate short-term and long-term moving averages.
    """
    stock_data['SMA_20'] = stock_data['close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['close'].rolling(window=50).mean()
    return stock_data

def generate_signals(stock_data):
    """
    Generate buy and sell signals based on moving averages crossover.
    """
    stock_data['Signal'] = 0  # 1 for Buy, -1 for Sell
    stock_data['Position'] = 0  # Current position: 1 for holding, 0 for not holding
    for i in range(1, len(stock_data)):
        if stock_data['SMA_20'].iloc[i] > stock_data['SMA_50'].iloc[i] and stock_data['SMA_20'].iloc[i-1] < stock_data['SMA_50'].iloc[i-1]:
            stock_data['Signal'].iloc[i] = 1  # Buy signal
        elif stock_data['SMA_20'].iloc[i] < stock_data['SMA_50'].iloc[i] and stock_data['SMA_20'].iloc[i-1] > stock_data['SMA_50'].iloc[i-1]:
            stock_data['Signal'].iloc[i] = -1  # Sell signal
    return stock_data

def calculate_performance_metrics(stock_name, stock_data):
    """
    Calculate performance metrics.
    """
    portfolio_returns = stock_data['Portfolio Returns'].dropna()  # Drop NA values for accurate calculations
    annualized_portfolio_returns = np.power((1 + portfolio_returns).prod(), (245 * 25) / len(portfolio_returns)) - 1  # Assuming 245 trading days in a year and 25 15-min windows per days(Indian stock market timings - 9.15 to 3.30)

    portfolio_volatility = portfolio_returns.std() * np.sqrt(245 * 25)

    risk_free_rate = 0.06  # Assuming a 6% risk-free rate, adjust as necessary
    sharpe_ratio = (annualized_portfolio_returns - risk_free_rate) / portfolio_volatility

    # Maximum drawdown calculation based on 'Portfolio Value'
    rolling_max = stock_data['Portfolio Value'].cummax()
    daily_drawdown = stock_data['Portfolio Value'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()

    metrics = {
        'Stock Name': stock_name,
        'Annualized Portfolio Returns': annualized_portfolio_returns,
        'Portfolio Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown
    }

    return metrics

def calculate_volatility(stock_data, window=20):
    """Calculate rolling volatility using standard deviation of returns."""
    stock_data['Daily Returns'] = stock_data['close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily Returns'].rolling(window=window).std() * np.sqrt(245 * 25)  # Adjusted for 15-minute intervals
    return stock_data

def adjust_position_size(stock_data, initial_capital, fixed_fraction=0.1):
    """
    Adjust position size based on current volatility.
    """
    # Calculate volatility
    stock_data = calculate_volatility(stock_data)

    # Use the most recent volatility for adjusting position size
    current_volatility = stock_data['Volatility'].iloc[-1]  # Most recent volatility

    # Normalize volatility to adjust position size inversely
    # Here, we're simply using the inverse of current volatility, scaled by a factor for manageability.
    normalized_volatility = 1 / current_volatility
    normalized_volatility_scaled = normalized_volatility / stock_data['Volatility'].max()

    # Calculate position size based on the fixed fraction of capital and normalized volatility
    stock_data['Position Size'] = initial_capital * fixed_fraction * normalized_volatility_scaled

    return stock_data

def simulate_trades(stock_data, initial_capital):
    """Simulate trades with dynamic position sizing and update portfolio value."""
    capital = initial_capital

    # Initialize 'Portfolio Value' and 'Invested Capital' columns
    stock_data['Portfolio Value'] = np.nan
    stock_data['Invested Capital'] = np.nan

    # Set initial portfolio value
    stock_data['Portfolio Value'].iloc[0] = initial_capital
    stock_data['Invested Capital'].iloc[0] = 0

    for i in range(1, len(stock_data)):
        if stock_data['Signal'].iloc[i] == 1:  # Buy signal
            num_shares = capital / stock_data['close'].iloc[i]  # Assuming full capital allocation
            cost = num_shares * stock_data['close'].iloc[i]
            if cost <= capital:  # Ensure we have enough capital
                capital -= cost
                stock_data['Invested Capital'].iloc[i] = cost
        elif stock_data['Signal'].iloc[i] == -1:  # Sell signal
            capital += stock_data['Invested Capital'].iloc[i-1]  # Add back the invested capital from previous period
            stock_data['Invested Capital'].iloc[i] = 0

        # Update 'Portfolio Value'
        stock_data['Portfolio Value'].iloc[i] = capital + (stock_data['Invested Capital'].iloc[i] if pd.notna(stock_data['Invested Capital'].iloc[i]) else 0)

    # Forward fill 'Invested Capital' for periods without trades
    stock_data['Invested Capital'].ffill(inplace=True)

    # Calculate returns based on portfolio value after ensuring all values are filled
    stock_data['Portfolio Returns'] = stock_data['Portfolio Value'].pct_change()

    return stock_data

def update_performance_metrics_with_portfolio(stock_name, stock_data):
    """Update performance metrics calculation to include portfolio-based metrics."""
    metrics = calculate_performance_metrics(stock_name, stock_data)  # Assuming this function is adapted to use portfolio returns
    metrics['Final Portfolio Value'] = stock_data['Portfolio Value'].iloc[-1]
    return metrics

def run_strategy_with_enhancements(data, stock_name, initial_capital=100000):
    """Run the enhanced strategy for a given stock name."""
    stock_data = load_and_prepare_data(data, stock_name)
    stock_data = calculate_moving_averages(stock_data)
    stock_data = generate_signals(stock_data)
    stock_data = adjust_position_size(stock_data, initial_capital)
    stock_data = simulate_trades(stock_data, initial_capital)
    metrics = update_performance_metrics_with_portfolio(stock_name, stock_data)
    return metrics

def run_all_stocks_with_enhancements(file_path, initial_capital=100000):
    """
    Run the complete enhanced strategy for all stocks present in data.
    """
    allstock_data = pd.read_csv(file_path)
    allstock_names = allstock_data.stock_name.unique()
    allstock_metrics = []

    for curr_stock in allstock_names:
        curr_metrics = run_strategy_with_enhancements(allstock_data, curr_stock, initial_capital)
        allstock_metrics.append(curr_metrics)
    result_metrics = pd.DataFrame(allstock_metrics)
    return result_metrics

# Example usage
file_path = 'data/nifty50_historicalData.csv'
result_path = 'data/nifty50_enhanced_returns_metrics.csv'
final_output = run_all_stocks_with_enhancements(file_path)
final_output.to_csv(result_path)
