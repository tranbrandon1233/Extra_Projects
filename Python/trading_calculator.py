import pandas as pd
from datetime import datetime
from io import StringIO
import pytz
from dataclasses import dataclass
from typing import Optional, List
from collections import defaultdict
import logging

# Configure logging to capture errors and warnings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class OptionsTradeMetrics:
    symbol: str
    trade_duration: float  # in hours
    entry_price: float
    exit_price: float
    best_exit_price: float
    position_mfe: float
    position_mae: float
    actual_roi: float
    potential_roi: float
    lost_alpha: float
    efficiency_score: float
    risk_adjusted_return: float
    position_mfe_mae_ratio: float

    def __repr__(self):
        return (f"OptionsTradeMetrics(symbol={self.symbol}, trade_duration={self.trade_duration:.2f}h, "
                f"entry_price={self.entry_price}, exit_price={self.exit_price}, "
                f"best_exit_price={self.best_exit_price}, position_mfe={self.position_mfe}, "
                f"position_mae={self.position_mae}, actual_roi={self.actual_roi:.2f}%, "
                f"potential_roi={self.potential_roi:.2f}%, lost_alpha={self.lost_alpha:.2f}, "
                f"efficiency_score={self.efficiency_score:.2f}%, "
                f"risk_adjusted_return={self.risk_adjusted_return:.2f}, "
                f"position_mfe_mae_ratio={self.position_mfe_mae_ratio:.2f})")

def calculate_metrics(row, quantity=100) -> Optional[OptionsTradeMetrics]:
    """
    Calculate advanced performance metrics for a single options trade.

    Parameters:
    - row: A pandas Series containing trade data.
    - quantity: The number of contracts traded. Defaults to 100.

    Returns:
    - An instance of OptionsTradeMetrics with calculated metrics, or None if parsing fails.
    """
    est = pytz.timezone('US/Eastern')
    utc = pytz.utc

    required_columns = ['Open Date', 'Open Time', 'Close Date', 'Close Time',
                        'Best Exit Time', 'Avg Buy Price', 'Avg Sell Price',
                        'Best Exit Price', 'Position MFE', 'Position MAE', 'Symbol']

    # Check for the presence of required columns
    missing_columns = [col for col in required_columns if col not in row or pd.isna(row[col])]
    if missing_columns:
        logging.warning(f"Missing columns {missing_columns} for Symbol {row.get('Symbol', 'Unknown')}. Skipping row.")
        return None

    try:
        # Parse Open Time
        open_time_str = row['Open Date'] + ' ' + row['Open Time']
        # Remove timezone abbreviations if present
        open_time_str_clean = open_time_str.replace('EST', '').replace('EDT', '').strip()
        open_time = est.localize(datetime.strptime(open_time_str_clean, '%Y-%m-%d %H:%M:%S')).astimezone(utc)
        
        # Parse Close Time
        close_time_str = row['Close Date'] + ' ' + row['Close Time']
        close_time_str_clean = close_time_str.replace('EST', '').replace('EDT', '').strip()
        close_time = est.localize(datetime.strptime(close_time_str_clean, '%Y-%m-%d %H:%M:%S')).astimezone(utc)
        
        # Parse Best Exit Time
        best_exit_time_str = row['Best Exit Time']
        try:
            # Attempt to parse with timezone information
            best_exit_time = datetime.strptime(best_exit_time_str, '%Y-%m-%d %H:%M:%S %Z')
            # Verify if the timezone is UTC; if not, convert to UTC
            if best_exit_time.tzinfo != utc:
                best_exit_time = best_exit_time.astimezone(utc)
        except ValueError:
            # If timezone info is missing or incorrect, assume UTC
            best_exit_time = pytz.utc.localize(datetime.strptime(best_exit_time_str, '%Y-%m-%d %H:%M:%S'))
        
    except ValueError as e:
        logging.error(f"Error parsing date/time for Symbol {row['Symbol']}: {e}")
        return None

    # Calculate Trade Duration in hours
    trade_duration = (close_time - open_time).total_seconds() / 3600

    # Extract Prices
    entry_price = row['Avg Buy Price']
    exit_price = row['Avg Sell Price']
    best_exit_price = row['Best Exit Price']

    # Calculate P&L
    actual_pnl = (exit_price - entry_price) * quantity
    potential_pnl = (best_exit_price - entry_price) * quantity
    lost_alpha = potential_pnl - actual_pnl

    # Calculate ROI (Return on Investment)
    actual_roi = (actual_pnl / (entry_price * quantity)) * 100 if entry_price != 0 else 0
    potential_roi = (potential_pnl / (entry_price * quantity)) * 100 if entry_price != 0 else 0

    # Calculate Efficiency Score
    denominator = (best_exit_price - entry_price)
    if denominator != 0:
        efficiency_score = ((exit_price - entry_price) / denominator) * 100
    else:
        efficiency_score = 0  # Avoid division by zero

    # Position MFE/MAE Ratio
    position_mfe = row['Position MFE']
    position_mae = row['Position MAE']
    if position_mae != 0:
        position_mfe_mae_ratio = position_mfe / abs(position_mae)
    else:
        position_mfe_mae_ratio = 0

    # Calculate Risk-Adjusted Return
    if (trade_duration * abs(position_mae)) != 0:
        risk_adjusted_return = actual_pnl / (trade_duration * abs(position_mae))
    else:
        risk_adjusted_return = 0

    return OptionsTradeMetrics(
        symbol=row['Symbol'],
        trade_duration=trade_duration,
        entry_price=entry_price,
        exit_price=exit_price,
        best_exit_price=best_exit_price,
        position_mfe=position_mfe,
        position_mae=position_mae,
        actual_roi=actual_roi,
        potential_roi=potential_roi,
        lost_alpha=lost_alpha,
        efficiency_score=efficiency_score,
        risk_adjusted_return=risk_adjusted_return,
        position_mfe_mae_ratio=position_mfe_mae_ratio
    )

def group_patterns_by_duration(patterns: List[OptionsTradeMetrics], tolerance_sec: int = 5) -> defaultdict:
    """
    Group detected trade patterns by similar durations within a specified tolerance.

    Parameters:
    - patterns: A list of OptionsTradeMetrics instances.
    - tolerance_sec: The tolerance in seconds for grouping durations.

    Returns:
    - A defaultdict with duration ranges as keys and lists of patterns as values.
    """
    grouped = defaultdict(list)
    for pattern in patterns:
        # Convert trade_duration from hours to seconds
        duration_sec = pattern.trade_duration * 3600
        # Determine the lower bound of the group
        group_key = int(duration_sec // tolerance_sec) * tolerance_sec
        grouped[group_key].append(pattern)
    return grouped

def generate_optimal_exit_suggestions(patterns: List[OptionsTradeMetrics]) -> List[OptionsTradeMetrics]:
    """
    Generate optimal exit suggestions based on historical patterns.
    This is a placeholder function and can be enhanced with more sophisticated logic.

    Parameters:
    - patterns: A list of OptionsTradeMetrics instances.

    Returns:
    - A list of OptionsTradeMetrics instances representing optimal exit suggestions.
    """
    # Placeholder: Return the patterns as optimal suggestions
    # Implement statistical analysis or machine learning models here
    return patterns

def calculate_aggregate_statistics(patterns: List[OptionsTradeMetrics]) -> pd.DataFrame:
    """
    Calculate aggregate statistics by symbol.

    Parameters:
    - patterns: A list of OptionsTradeMetrics instances.

    Returns:
    - A pandas DataFrame containing aggregate statistics.
    """
    if not patterns:
        return pd.DataFrame()

    data = {
        'Symbol': [p.symbol for p in patterns],
        'Lost Alpha': [p.lost_alpha for p in patterns],
        'Efficiency Score': [p.efficiency_score for p in patterns],
        'Risk-Adjusted Return': [p.risk_adjusted_return for p in patterns]
    }
    df = pd.DataFrame(data)
    aggregate_stats = df.groupby('Symbol').agg({
        'Lost Alpha': 'sum',
        'Efficiency Score': 'mean',
        'Risk-Adjusted Return': 'mean'
    }).reset_index()
    return aggregate_stats

def main():
    """
    Main function to execute the options trading data analysis.
    """
    data = """Account Name,Adjusted Cost,Adjusted Proceeds,Avg Buy Price,Avg Sell Price,Exit Efficiency,Best Exit,Best Exit Price,Best Exit Time,Close Date,Close Time,Commission,Custom Tags,Duration,Entry Price,Executions,Exit Price,Gross P&L,Trade Risk,Initial Target,Instrument,Spread Type,Market Cap,Mistakes,Net P&L,Net ROI,Open Date,Open Time,Pips,Reward Ratio,Playbook,Points,Position MAE,Position MFE,Price MAE,Price MFE,Realized RR,Return Per Pip,Reviewed,Sector,Setups,Side,Status,Symbol,Ticks Value,Ticks Per Contract,Fee,Swap,Rating,Quantity,Zella Score
Robinhood 0281,11291.0,15041.6,8.065,10.744,112.21,3342.0,12.27,2022-11-09 20:25:00 UTC,2022-11-09,10:26:49 EST,0.0,"",168245.717,11.51,27,8.19,3750.0,0.0,0.0,2022-11-09 276 PUT,single,64891144500.0,"",3750.0,0.33212293,2022-11-07,11:42:44 EST,,,"",,-2957.0,0.0,4.28,10.33,,,false,No sector,"",short,Win,QQQ,,,0.0,0.0,,14.0,126.81772066
Robinhood 0281,19357.8,21463.2,8.799,9.756,102.18,2061.0,9.26,2022-11-07 16:36:00 UTC,2022-11-07,10:42:28 EST,0.0,"",265379.0,9.85,43,8.81,2106.0,0.0,0.0,2022-11-18 269 PUT,single,64891144500.0,"",2106.0,0.10879785,2022-11-04,09:59:29 EDT,,,"",,-298.0,518.0,7.98,12.06,,,false,No sector,"",short,Win,QQQ,,,0.0,0.0,,22.0,406.56370656"""

    # Read the CSV data
    try:
        data_io = StringIO(data)
        df = pd.read_csv(data_io)
    except Exception as e:
        logging.error(f"Error reading CSV data: {e}")
        return

    # Check for essential columns before processing
    essential_columns = ['Open Date', 'Open Time', 'Close Date', 'Close Time',
                        'Best Exit Time', 'Avg Buy Price', 'Avg Sell Price',
                        'Best Exit Price', 'Position MFE', 'Position MAE', 'Symbol']
    missing_essential = [col for col in essential_columns if col not in df.columns]
    if missing_essential:
        logging.error(f"Missing essential columns in data: {missing_essential}")
        return

    results = []
    for index, row in df.iterrows():
        metrics = calculate_metrics(row, quantity=row.get('Quantity', 100))
        if metrics:
            results.append(metrics)
            print(metrics)
        else:
            logging.warning(f"Failed to calculate metrics for row index {index}.")

    if not results:
        print("No price reversal patterns detected.")
        return

    # Group patterns by similar durations within a 5-second range
    grouped_patterns = group_patterns_by_duration(results, tolerance_sec=5)

    print("\nGrouped Patterns by Duration (within 5-second ranges):")
    for group_key, patterns in grouped_patterns.items():
        duration_range = f"{group_key}-{group_key + 5} seconds"
        print(f"\nDuration Range: {duration_range}")
        for pattern in patterns:
            print(pattern)

    # Calculate aggregate statistics by symbol
    aggregate_stats = calculate_aggregate_statistics(results)
    if not aggregate_stats.empty:
        print("\nAggregate Statistics by Symbol:")
        print(aggregate_stats)
    else:
        print("No aggregate statistics to display.")

    # Generate optimal exit suggestions (placeholder)
    optimal_exits = generate_optimal_exit_suggestions(results)
    print("\nOptimal Exit Suggestions:")
    for exit_suggestion in optimal_exits:
        print(exit_suggestion)

if __name__ == "__main__":
    main()