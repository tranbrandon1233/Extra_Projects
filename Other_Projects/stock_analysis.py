import pandas as pd
import numpy as np
import datetime as dt
# from pkgutil import get_data
import csv
from matplotlib import pyplot as plt

class ManualStrategy:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def bollinger_bands(self,symbol, dates, window=20):

        # stock_data = get_data([symbol], dates)
        # prices = stock_data[symbol]

        prices = []
        with open('stocks.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if symbol in row[0]:
                    datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
                    if datetime in dates:
                        prices.append(row[2])
        prices = pd.Series(prices)
        middle_band = prices.rolling(window=window).mean()

        rolling_std = prices.rolling(window=window).std()

        upper_band = middle_band + (rolling_std * 2)
        lower_band = middle_band - (rolling_std * 2)

        bands = pd.DataFrame(index=prices.index)
        bands['Upper Band'] = upper_band
        bands['Middle Band'] = middle_band
        bands['Lower Band'] = lower_band

        return bands

    def plot_bollinger_bands(self,bands, symbol, dates, filename = 'bollinger_bands.png'):

        #stock_data = get_data([symbol], dates)
        #prices = stock_data[symbol]

        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)

        plt.figure(figsize=(14, 7))
        plt.plot(prices.index, prices, label=f"{symbol} Price")
        plt.plot(bands.index, bands['Upper Band'], label='Upper Band', linestyle='--')
        plt.plot(bands.index, bands['Middle Band'], label='Middle Band', linestyle='-')
        plt.plot(bands.index, bands['Lower Band'], label='Lower Band', linestyle='--')
        plt.title(f"Bollinger Bands for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(f"images/{filename}")
        plt.close()

    def ema(self,symbol, dates, window=20):

        # stock_data = get_data(symbol, dates)
        # prices = stock_data[symbol]
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)

        ema = prices.ewm(span=window, adjust=False).mean()

        return ema

    def plot_ema(self,ema, symbol, dates, window = 20, filename = 'ema.png'):
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)

        # stock_data = get_data([symbol], dates)
        #prices = stock_data[symbol]

        plt.figure(figsize=(14, 7))
        plt.plot(prices.index, prices, label=f"{symbol} Price")
        plt.plot(ema.index, ema, label=f'{window}-day EMA', linestyle='--')
        plt.title(f"Exponential Moving Average (EMA) for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(f"images/{filename}")
        plt.close()


    def rsi(self,symbol, dates, window=14):

        #stock_data = get_data([symbol], dates)
        #prices = stock_data[symbol]
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)

        delta = prices.diff().dropna()

        gain, loss = delta.copy(), delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = -loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


    def plot_rsi(self,rsi, symbol, dates, window = 14, filename = 'rsi.png'):


        #stock_data = get_data(symbol, dates)
        #prices = stock_data[symbol]
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)

        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(prices.index, prices, label=f"{symbol} Price", color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('RSI', color=color)
        ax2.plot(rsi.index, rsi, label=f'{window}-day RSI', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(70, color=color, linestyle='--', label="Overbought Threshold(70)")
        ax2.axhline(30, color=color, linestyle='--', label="Oversold Threshold(30)")
        ax2.set_yticks(range(0, 101, 10))

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.tight_layout()
        plt.title(f"Price and RSI for {symbol}")
        plt.savefig(f"images/{filename}")
        plt.close()

    def rate_of_change(self,symbol, dates, window=14):
        #stock_data = get_data([symbol], dates)
        #prices = stock_data[symbol]

        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)
        roc = ((prices - prices.shift(window)) / prices.shift(window)) * 100
        return roc

    def plot_roc(self,roc, symbol, dates, window = 14, filename = 'roc.png'):

        # stock_data = get_data([symbol], dates)
        # prices = stock_data[symbol]
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)

        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(prices.index, prices, label=f"{symbol} Price", color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('ROC (%)', color=color)
        ax2.plot(roc.index, roc, label=f'{window}-day ROC', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(0, color=color, linestyle='--', label="Threshold(0)")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.title(f"Price and Rate of Change (ROC) for {symbol}")
        plt.savefig(f"images/{filename}")
        plt.close()

    def macd(self,symbol, dates, short_window=12, long_window=26, signal_window=9):

        #stock_data = get_data([symbol], dates)
        #prices = stock_data[symbol]
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)

        ema_short = prices.ewm(span=short_window, adjust=False).mean()
        ema_long = prices.ewm(span=long_window, adjust=False).mean()

        macd_line = ema_short - ema_long

        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        macd = pd.DataFrame(index=prices.index)
        macd['MACD Line'] = macd_line
        macd['Signal Line'] = signal_line

        return macd

    def plot_macd(self,macd, symbol, dates, short_window=12, long_window=26, signal_window=9, filename = 'macd.png'):

        # stock_data = get_data([symbol], dates)
        # prices = stock_data[symbol]
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)
        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(prices.index, prices, label=f"{symbol} Price", color=color, linewidth=0.75)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('MACD', color=color)
        ax2.plot(macd.index, macd['MACD Line'], label='MACD Line', color=color)
        ax2.plot(macd.index, macd['Signal Line'], label='Signal Line', linestyle='--', color='tab:orange')
        ax2.bar(macd.index, macd['MACD Line'] - macd['Signal Line'], label='Histogram', color='grey', alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        plt.title(f"MACD for {symbol} (Short: {short_window}, Long: {long_window}, Signal: {signal_window})")

        plt.savefig(f"images/{filename}")
        plt.close()

    def run(self,symbol,sd,ed):
        dates = pd.date_range(sd,ed)
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        prices = pd.Series(prices)
        dates = pd.date_range(sd,ed)
        bands = self.bollinger_bands(symbol,dates)

        self.plot_bollinger_bands(bands, symbol, dates, filename='bollinger_bands.png')

        ema_values = self.ema(symbol, dates)
        self.plot_ema(ema_values, symbol, dates)

        rsi_values = self.rsi(symbol, dates)
        self.plot_rsi(rsi_values, symbol, dates)

        roc_values = self.rate_of_change(symbol, dates)
        self.plot_roc(roc_values, symbol, dates)

        macd_values = self.macd(symbol, dates)
        self.plot_macd(macd_values, symbol, dates)

    def testPolicy(self, symbol='IBM', sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000):
        dates = pd.date_range(sd, ed)
        # prices_all = get_data(symbol, dates)  # automatically adds SPY
        # prices = prices_all[symbol]  # only portfolio symbols
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if symbol in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in dates:
                prices.append(float(row[2]))
        
        prices = pd.Series(prices)

        # Get the indicators
        bollinger = self.get_bollinger(symbol_str=symbol,date_window=dates)
        rsi = self.get_rsi(prices)
        macd = self.get_macd(prices)

        # Trading signals (-1, 0, 1)
        signals = self.generate_signals(bollinger, rsi, macd)

        # Generate trades from these signals
        trades = self.generate_trades(signals, prices)

        if self.verbose:
            print(trades)
        return trades

    def get_bollinger(self,symbol_str,date_window,window=20):
        prices = []
        with open('stocks.csv', 'r') as file:
          reader = csv.reader(file)
          for row in reader:
            if str(symbol_str) in row[0]:
              datetime = dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
              if datetime in date_window:
                prices.append(float(row[2]))
        prices = pd.Series(prices)
        middle_band = prices.rolling(window=int(window)).mean()

        rolling_std = prices.rolling(window=int(window)).std()

        bb_value = (prices - middle_band) / (2 * rolling_std)
        return bb_value

    def get_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_macd(self, prices, short_window=12, long_window=26, signal_window=9):
        ema_short = prices.ewm(span=short_window, adjust=False).mean()
        ema_long = prices.ewm(span=long_window, adjust=False).mean()

        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        macd_value = macd_line - signal_line
        return macd_value

    def generate_signals(self, bollinger, rsi, macd):
        signals = pd.Series(index=[i for i in range(len(bollinger))], data=bollinger)

        # Generate signals based on Bollinger Bands, RSI, and MACD
        signal_bb = (bollinger > 1).astype(int) - (bollinger < -1).astype(int)
        signal_rsi = (rsi > 70).astype(int) - (rsi < 30).astype(int)
        signal_macd = (macd > 0).astype(int) - (macd < 0).astype(int)

        signals = signal_bb + signal_rsi + signal_macd
        return signals

    def generate_trades(self, signals, prices):
        trades = pd.DataFrame(data=np.zeros(len(signals)), index=signals.index, columns=[prices.name])
        current_pos = 0

        for i in range(1, len(signals)):
            if signals[i] > 0 and current_pos == 0:  # Buy signal
                trades.iloc[i] = 1000
                current_pos = 1000
            elif signals[i] < 0 and current_pos == 0:  # Sell signal
                trades.iloc[i] = -1000
                current_pos = -1000
            elif signals[i] > 0 and current_pos == -1000:  # Exit short & go long
                trades.iloc[i] = 2000
                current_pos = 1000
            elif signals[i] < 0 and current_pos == 1000:  # Exit long & go short
                trades.iloc[i] = -2000
                current_pos = -1000
        return trades

if __name__ == "__main__":
    ms = ManualStrategy(verbose=True)
    sd=dt.datetime(2008, 1, 1)
    ed=dt.datetime(2008, 5, 1)
    # df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2008, 5, 1), sv=100000)
    ms.run(symbol="JPM",sd=sd,ed=ed)
    print(df_trades[df_trades != 0].dropna())