import pandas as pd

def load_and_resample_data(filepath, date_column, target_column, freq='M'):
    df = pd.read_csv(filepath, parse_dates=[date_column], index_col=date_column)
    resampled_data = df[target_column].resample(freq).sum()
    return resampled_data
