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

def update_position(stock_data):
    """
    Update the position based on signals.
    """
    for i in range(1, len(stock_data)):
        if stock_data['Signal'].iloc[i] == 1:
            stock_data['Position'].iloc[i] = 1
        elif stock_data['Signal'].iloc[i] == -1:
            stock_data['Position'].iloc[i] = 0
        else:
            stock_data['Position'].iloc[i] = stock_data['Position'].iloc[i-1]
    return stock_data

def calculate_returns(stock_data):
    """
    Calculate stock and strategy returns, including cumulative returns.
    """
    stock_data['Stock Returns'] = stock_data['close'].pct_change()
    stock_data['Strategy Returns'] = stock_data['Stock Returns'] * stock_data['Position'].shift(1)
    stock_data['Cumulative Stock Returns'] = (1 + stock_data['Stock Returns']).cumprod()
    stock_data['Cumulative Strategy Returns'] = (1 + stock_data['Strategy Returns']).cumprod()
    return stock_data

def calculate_performance_metrics(stock_name, stock_data):
    """
    Calculate additional performance metrics like annualized returns, volatility, Sharpe ratio, and maximum drawdown.
    """
    # Adjust the calculation for annualized returns and volatility
    annualized_strategy_returns = np.power((1 + stock_data['Strategy Returns']).prod(), (245 * 25) / len(stock_data)) - 1 # Assuming 245 trading days in a year and 25 15-min windows per days(Indian stock market timings - 9.15 to 3.30)
    annualized_stock_returns = np.power((1 + stock_data['Stock Returns']).prod(), (245 * 25) / len(stock_data)) - 1

    strategy_volatility = stock_data['Strategy Returns'].std() * np.sqrt(245*25)
    stock_volatility = stock_data['Stock Returns'].std() * np.sqrt(245*25)

    risk_free_rate = 0.06  # Assuming a 6% risk-free rate
    sharpe_ratio = (annualized_strategy_returns - risk_free_rate) / strategy_volatility

    rolling_max = stock_data['Cumulative Strategy Returns'].cummax()
    daily_drawdown = stock_data['Cumulative Strategy Returns'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()

    metrics = {
        'Stock Name' : stock_name,
        'Annualized Strategy Returns': annualized_strategy_returns,
        'Annualized Stock Returns': annualized_stock_returns,
        'Strategy Volatility': strategy_volatility,
        'Stock Volatility': stock_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown
    }

    return metrics

def run_strategy(data, stock_name):
    """
    Run the complete trading strategy for a given stock name.
    """
    stock_data = load_and_prepare_data(data, stock_name)
    stock_data = calculate_moving_averages(stock_data)
    stock_data = generate_signals(stock_data)
    stock_data = update_position(stock_data)
    stock_data = calculate_returns(stock_data)
    metrics = calculate_performance_metrics(stock_name, stock_data)
    return metrics

def run_all_stocks(file_path):
    """
    Run the complete code for all stocks present in data.
    """
    result_df = pd.DataFrame()
    allstock_metrics = []
    allstock_data = pd.read_csv(file_path)
    allstock_names = allstock_data.stock_name.unique()
    for curr_stock in allstock_names:
        curr_metrics = run_strategy(allstock_data, curr_stock)
        allstock_metrics.append(curr_metrics)
    result_metrics = pd.DataFrame(allstock_metrics)
    final_output = pd.concat([result_df, result_metrics], ignore_index=True)

    return final_output

# Example usage
file_path = 'data/nifty50_historicalData.csv'  # Update with the correct path
result_path = 'data/nifty50_returns_metrics.csv'
final_output = run_all_stocks(file_path)
final_output.to_csv(result_path)
