from shared_utils.data import get_securities_data
from tti.indicators import *

if __name__ == '__main__':
    securities = ['GOOG', 'AAPL', 'TSLA', 'SPY']
    data = get_securities_data(securities)

    goog_data = data['GOOG']

    adl_indicator = AccumulationDistributionLine(input_data=goog_data)

    simulation_data, simulation_statistics, simulation_graph = \
        adl_indicator.getTiSimulation(
            close_values=goog_data[['close']])

    print(simulation_data)





