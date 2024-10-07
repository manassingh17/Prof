from src.data_preprocessing import load_and_resample_data
from src.stationarity_check import check_stationarity
from src.model_training import train_arima_model, train_model_with_order
from src.forecasting import forecast_and_plot

# Load and resample data
data = load_and_resample_data('Pakistan Largest Ecommerce Dataset.csv', 'created_at', 'grand_total')

# Check stationarity
is_stationary, stationarity_result = check_stationarity(data)
print("ADF Statistic:", stationarity_result[0])
print("p-value:", stationarity_result[1])
print("Stationary:", is_stationary)

# If not stationary, difference the data
if not is_stationary:
    data = data.diff().dropna()

# Train model
auto_model = train_arima_model(data)
model_fit = train_model_with_order(data, auto_model.order)

# Forecast and plot results
forecast_and_plot(model_fit, data)
