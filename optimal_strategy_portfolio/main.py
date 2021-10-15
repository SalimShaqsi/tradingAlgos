from shared_utils.data import get_data

symbols = ['GOOG', 'TSLA', 'AMZN', 'FB', 'SPY']

train_start_date = '2018-01-01'
train_end_date = '2020-01-01'

test_start_date = train_end_date
test_end_date = '2020-02-01'

data = get_data(symbols)

