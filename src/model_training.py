from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def train_arima_model(timeseries, seasonal=False):
    auto_model = auto_arima(timeseries, seasonal=seasonal, stepwise=True, 
                            trace=True, error_action='ignore', suppress_warnings=True)
    return auto_model

def train_model_with_order(timeseries, order):
    model = ARIMA(timeseries, order=order)
    model_fit = model.fit()
    return model_fit
