import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
import numpy as np

start_date = '2014-01-01'
end_date = '2020-01-01'
SRC_DATA_FILENAME = 'spy_data.pkl'

try:
    goog_data2 = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    goog_data2 = data.DataReader('SPY', 'yahoo', start_date, end_date)
    goog_data2.to_pickle(SRC_DATA_FILENAME)

goog_data = goog_data2.tail(620)
lows = goog_data['Low']
highs = goog_data['High']

resistance = lows.head(200).max()
support = lows.head(200).min()

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Google prices in $')
highs.plot(ax=ax1, color='c')
lows.plot(ax=ax1, color='y')
plt.hlines(resistance, highs.price_index.values[0], highs.price_index.values[-1], color='g')
plt.hlines(support, lows.price_index.values[0], lows.price_index.values[-1], color='r')
plt.axvline(linewidth=2, color='n', x=lows.price_index.values[200], linestyle=':')
plt.show()


def support_resistance_trading(data, window_size=50, tolerance=0.2, count_limit=2):
    data['support_tolerance'] = 0
    data['resistance_tolerance'] = 0
    data['support'] = 0
    data['resistance'] = 0
    data['support_count'] = 0
    data['resistance_count'] = 0
    data['position'] = 0
    data['signal_arr'] = 0
    data.fillna(0)
    for i in range(window_size, len(data)):
        data_section = data[i - window_size:i]
        data['support'][i] = data_section['price'].min()
        data['resistance'][i] = data_section['price'].max()
        tol = (data['resistance'][i] - data['support'][i]) * tolerance
        data['support_tolerance'][i] = data['support'][i] + tol
        data['resistance_tolerance'][i] = data['resistance'][i] - tol
        data['support_count'][i] = \
            (data['support_count'][i - 1] + 1 or 1) if data['price'][i] < data['support_tolerance'][i] else 0
        data['resistance_count'][i] = \
            (data['resistance_count'][i - 1] + 1 or 1) if data['price'][i] > data['resistance_tolerance'][i] else 0

        #print(data['support_count'][i], data['resistance_count'][i], data['position'][i-1])

        invested = data['position'][i-1] > 0
        #print(invested)
        if data['support_count'][i] >= count_limit and not invested:
            data['signal_arr'][i] = 1
        elif data['resistance_count'][i] >= count_limit and invested:
            data['signal_arr'][i] = -1

        data['position'][i] = (data['position'][i-1] or 0) + data['signal_arr'][i] * data['price'][i]
        #print(data['signal_arr'][i],  data['position'][i])

    return data

goog_data['price'] = goog_data['Adj Close']
data = support_resistance_trading(goog_data, window_size=10, tolerance=0.3, count_limit=2)
data['avg_cash'] = 1000 - data['position']
data['avg_cash'].plot()
data['position'].plot()
plt.show()

r = (data['avg_cash'][-1] - data['avg_cash'][0]) / data['avg_cash'][0]
baseline_r = (data['price'][-1] - data['price'][0]) / data['price'][0]
print(baseline_r, r)
