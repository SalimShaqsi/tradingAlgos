from pandas_datareader import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# First day
start_date = '2014-01-01'
# Last day
end_date = '2018-01-01'
# Call the function DataReader from the class data
goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)

goog_data_signal = pd.DataFrame(index=goog_data.index)
goog_data_signal['price'] = goog_data['Adj Close']
goog_data_signal['daily_difference'] = goog_data_signal['price'].diff()
goog_data_signal['signal_arr'] = np.where(goog_data_signal['daily_difference'] > 0, 1.0, 0.0)
goog_data_signal['positions'] = goog_data_signal['signal_arr'].diff()


initial_capital = 1000.0
portfolio = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)

portfolio['positions'] = (goog_data_signal['signal_arr'] * goog_data_signal['price'])
portfolio['cash'] = initial_capital - (goog_data_signal['signal_arr'].diff() * goog_data_signal['price']).cumsum()
portfolio['total'] = portfolio['positions'] + portfolio['cash']
print(portfolio)
