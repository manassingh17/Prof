import matplotlib.pyplot as plt

def forecast_and_plot(model_fit, timeseries, steps=2):
    forecast = model_fit.get_forecast(steps=steps)
    forecast_ci = forecast.conf_int()

    plt.figure(figsize=(10, 6))
    plt.plot(timeseries.index, timeseries, label='Historical Sales', color='red')
    plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, color='blue', label='Forecasted Sales')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                     color='gray', alpha=0.3, label='Confidence Interval')
    plt.axhline(y=1e8, color='green', linestyle='--', label='Reference Line at 1e8')
    plt.title('Actual vs Forecasted Sales with Confidence Interval')
    plt.legend()
    plt.show()
