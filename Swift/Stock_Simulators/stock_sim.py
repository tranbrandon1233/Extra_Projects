import numpy as np
import pandas as pd
import yfinance as yf
import multiprocessing
import time

# Constants
ANNUAL_TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
COSTS = 0.0002

# Function to fetch historical data
def fetch_data(symbol, start_date):
    data = yf.download(symbol, start=start_date)
    return data

# Function to define the trading strategy
def define_strategy(data, sp500_data, short_window, long_window, relative_strength_window):
    # Calculate moving averages
    data['Short MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long MA'] = data['Close'].rolling(window=long_window).mean()

    # Generate buy/sell signals based on moving average crossovers
    data['Buy'] = np.where(data['Short MA'] > data['Long MA'], 1, 0)
    data['Sell'] = np.where(data['Short MA'] < data['Long MA'], -1, 0)

    # Calculate relative strength
    data['Relative Strength'] = data['Close'].pct_change(relative_strength_window) / sp500_data['Close'].pct_change(relative_strength_window)

    # Apply relative strength filtering for specific symbols
    symbols = ['AMZN', 'META', 'MSFT', 'NVDA', 'AAPL', 'OXY']
    if data.index.name in symbols:
        data.loc[data['Buy'] == 1, 'Buy'] = np.where(data['Relative Strength'] > 1, 1, 0)
        data.loc[data['Sell'] == -1, 'Sell'] = np.where(data['Relative Strength'] < 1, -1, 0)

    return data

# Function to backtest the strategy
def backtest_strategy(data):
    # Determine position based on buy/sell signals
    data['Position'] = 0
    data['Position'] = data['Buy'] + data['Sell']
    data['Position'] = data['Position'].replace(0, np.nan).ffill().fillna(0)

    # Calculate returns
    data['Market Returns'] = data['Close'].diff()
    data['Strategy Returns'] = data['Market Returns'] * data['Position'].shift()

    # Apply transaction costs
    data['Transaction Costs'] = data['Close'].shift() * data['Position'].diff().abs() * COSTS
    data['Strategy Returns'] = data['Strategy Returns'] - data['Transaction Costs']

    # Calculate cumulative returns
    data['Cumulative Market Returns'] = data['Market Returns'].cumsum().fillna(0)
    data['Cumulative Strategy Returns'] = data['Strategy Returns'].cumsum().fillna(0)

    return data

# Function to analyze the performance
def analyze_performance(data, symbol, start):
    # Calculate metrics
    annualized_return = data['Strategy Returns'].mean() * ANNUAL_TRADING_DAYS
    drawdown = data['Cumulative Strategy Returns'].cummax() - data['Cumulative Strategy Returns']
    max_drawdown = drawdown.max()

    # Calculate Sortino Ratio
    downside_returns = data['Strategy Returns'][data['Strategy Returns'] < 0]
    downside_deviation = downside_returns.std() * np.sqrt(ANNUAL_TRADING_DAYS)
    sortino_ratio = (annualized_return - RISK_FREE_RATE) / downside_deviation

    # Calculate total transaction costs
    total_transaction_costs = data['Transaction Costs'].sum()

    # Print results
    print(f"{symbol} {start}")
    print("Market: ", data['Cumulative Market Returns'].iloc[-1])
    print("Net Strategy Returns (after costs): ", data['Cumulative Strategy Returns'].iloc[-1] - total_transaction_costs)
    print("Maximum Drawdown: ", max_drawdown)
    print("Sortino Ratio: ", sortino_ratio)

# Function to export data to a CSV file
def export_to_csv(data, filename):
    data.to_csv(filename)
    print(f"Data exported to {filename}")

# Function to analyze the strategy with given parameters (used for multiprocessing)
def analyze_strategy(params):
    short_window, long_window, data, volatility_bracket, relative_strength_window, sp500_data = params
    data_copy = data[data['Volatility Bracket'] == volatility_bracket].copy()
    if not data_copy.empty:
        data_copy = define_strategy(data_copy, sp500_data, short_window, long_window, relative_strength_window)
        data_copy = backtest_strategy(data_copy)
        annualized_return = data_copy['Strategy Returns'].mean() * ANNUAL_TRADING_DAYS
        downside_deviation = data_copy['Strategy Returns'][data_copy['Strategy Returns'] < 0].std() * np.sqrt(ANNUAL_TRADING_DAYS)
        sortino_ratio = (annualized_return - RISK_FREE_RATE) / downside_deviation
        return sortino_ratio, short_window, long_window, volatility_bracket, relative_strength_window
    else:
        return float('-inf'), None, None, volatility_bracket, None

# Main function to execute the backtesting workflow
def main(symbol, start, number):
    # Parameters and data fetching
    start_date = start
    windows = [5, 15, 30, 100]
    relative_strength_windows = [9, 14, 21, 31, 49, 99] 
    data = fetch_data(symbol, start_date)
    sp500_data = yf.download('^GSPC', start=start_date)

    # Iterate through different volatility timeframes
    for windows in windows:
        # Iterate through different relative strength windows
        for relative_strength_window in relative_strength_windows:
            # Calculate volatility and define brackets
            data['Volatility'] = data['Close'].pct_change().rolling(window=windows).std()
            volatility_brackets = ['Low', 'Medium', 'High']
            volatility_thresholds = [data['Volatility'].quantile(number), data['Volatility'].quantile(1-number)]
            data['Volatility Bracket'] = pd.cut(data['Volatility'], bins=[-np.inf] + list(volatility_thresholds) + [np.inf], labels=volatility_brackets)

            # Initialize variables to store the best parameters for each volatility bracket
            max_sortino_ratios = {bracket: float('-inf') for bracket in volatility_brackets}
            best_short_windows = {bracket: None for bracket in volatility_brackets}
            best_long_windows = {bracket: None for bracket in volatility_brackets}
            best_relative_strength_windows = {bracket: None for bracket in volatility_brackets}

            # Generate parameter combinations
            params = [(short_window, long_window, data, volatility_bracket, relative_strength_window, sp500_data)
                      for volatility_bracket in volatility_brackets
                      for short_window in range(2, 352, 1)
                      for long_window in range(2, 352, 1)]

            # Use multiprocessing to speed up parameter optimization
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(analyze_strategy, params)

            # Find the best parameters for each volatility bracket
            for result in results:
                sortino_ratio, short_window, long_window, volatility_bracket, relative_strength_window = result
                if sortino_ratio > max_sortino_ratios[volatility_bracket]:
                    max_sortino_ratios[volatility_bracket] = sortino_ratio
                    best_short_windows[volatility_bracket] = short_window
                    best_long_windows[volatility_bracket] = long_window
                    best_relative_strength_windows[volatility_bracket] = relative_strength_window

            # Apply the best strategy for each volatility bracket
            for volatility_bracket in volatility_brackets:
                data.loc[data['Volatility Bracket'] == volatility_bracket, 'Position'] = 0
                if best_short_windows[volatility_bracket] is not None and best_long_windows[volatility_bracket] is not None and best_relative_strength_windows[volatility_bracket] is not None:
                    data.loc[data['Volatility Bracket'] == volatility_bracket, 'Short MA'] = data[data['Volatility Bracket'] == volatility_bracket]['Close'].rolling(window=best_short_windows[volatility_bracket]).mean()
                    data.loc[data['Volatility Bracket'] == volatility_bracket, 'Long MA'] = data[data['Volatility Bracket'] == volatility_bracket]['Close'].rolling(window=best_long_windows[volatility_bracket]).mean()
                    data.loc[data['Volatility Bracket'] == volatility_bracket, 'Buy'] = np.where(data[data['Volatility Bracket'] == volatility_bracket]['Short MA'] > data[data['Volatility Bracket'] == volatility_bracket]['Long MA'], 1, 0)
                    data.loc[data['Volatility Bracket'] == volatility_bracket, 'Sell'] = np.where(data[data['Volatility Bracket'] == volatility_bracket]['Short MA'] < data[data['Volatility Bracket'] == volatility_bracket]['Long MA'], -1, 0)
                    data.loc[data['Volatility Bracket'] == volatility_bracket, 'Position'] = data[data['Volatility Bracket'] == volatility_bracket]['Buy'] + data[data['Volatility Bracket'] == volatility_bracket]['Sell']
                    data.loc[data['Volatility Bracket'] == volatility_bracket, 'Position'] = data[data['Volatility Bracket'] == volatility_bracket]['Position'].replace(0, np.nan).ffill().fillna(0)

            # Backtest and analyze the overall strategy
            data = backtest_strategy(data)

            # Add columns for tracking parameters
            data['Short MA Volatility Bracket'] = data['Volatility Bracket'].map(best_short_windows)
            data['Long MA Volatility Bracket'] = data['Volatility Bracket'].map(best_long_windows)
            data['Relative Strength Window'] = data['Volatility Bracket'].map(best_relative_strength_windows)

            # Analyze and print performance
            analyze_performance(data, symbol, start)

            # Print parameters (for tracking progress)
            print(f"{windows} {relative_strength_window} {number}")

    # Print execution time
    end_time = time.time()
    print(f"Execution Time: {round(end_time - start_time, 2)} seconds")

# Example usage
if __name__ == "__main__":
    symbol = 'OXY'
    number = [0.15, 0.2, 0.25, 0.33]
    start = '2000-01-01'
    for number in number:
        main(symbol, start, number)