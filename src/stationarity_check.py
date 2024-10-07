from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    is_stationary = result[1] <= 0.05
    return is_stationary, result
