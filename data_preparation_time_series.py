import pandas as pd
import numpy as np

# Function to create sliding windows for the time series data
def create_sliding_window(data, window_size):
    X, y = [], []
    # Loop over the data to create sliding windows
    for i in range(len(data) - window_size):
        # Append the windowed data to X (input features)
        # and the next data point to y (output)
        # Note that we reverse the windowed data so it's in order t, t-1, t-2, ...
        X.append(data[i:i + window_size][::-1])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Function to prepare the time series data
def prepare_time_series_data(data, window_size, location_column, time_column, target_column):
    # Sort the data by Location ID and Timestamp
    data = data.sort_values([location_column, time_column])
    # Group the data by Location ID
    grouped_data = data.groupby(location_column)

    X, y = [], []
    # Loop over each group
    for _, group in grouped_data:
        # Extract the target time series data
        kwh_data = group[target_column].values
        # Create sliding windows for this group
        X_group, y_group = create_sliding_window(kwh_data, window_size)
        # Add the windows to our lists
        X.extend(X_group)
        y.extend(y_group)

    return np.array(X), np.array(y), grouped_data

if __name__ == '__main__':
    # Example usage
    data = pd.DataFrame({
        'Location ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'Timestamp': pd.date_range('2023-01-01', periods=5).tolist() * 2,
        'kwh/day': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155],
        'Population': [5000, 5100, 5150, 5200, 5250, 3000, 3100, 3200, 3300, 3350],
        'Avg Income': [60000, 60500, 61000, 61200, 61500, 80000, 80500, 81000, 81500, 82000],
        'Avg Household Size': [6, 6, 6, 7, 6, 8, 7, 8, 8, 9]
    })\

    WINDOW_SIZE = 2
    TARGET_COLUMN = 'kwh/day'
    LOCATION_COLUMN = 'Location ID'
    TIME_COLUMN = 'Timestamp'

    # fetch feature names, that is all columns except the target column and the time column and the location column
    feature_names = [col for col in data.columns if col not in [TARGET_COLUMN, TIME_COLUMN, LOCATION_COLUMN]]

    # Prepare the data
    X, y, grouped_data = prepare_time_series_data(data, WINDOW_SIZE, LOCATION_COLUMN, TIME_COLUMN, TARGET_COLUMN)


    # Generate dynamic column names based on the window size
    column_names = [[LOCATION_COLUMN, TIME_COLUMN, TARGET_COLUMN + '(t+1)'] + [f'{TARGET_COLUMN}(t-{i})' for i in range(WINDOW_SIZE)] + feature_names]

    # Initialize an empty DataFrame with the dynamic column names
    processed_data = pd.DataFrame(columns=column_names)

    idx = 0
    # Loop over each group
    for loc_id, group in grouped_data:
        # Loop over each data point in the group
        for i in range(group.shape[0] - WINDOW_SIZE):
            # Extract the timestamp for t+1
            timestamp = group.iloc[i + WINDOW_SIZE][TIME_COLUMN]
            # Extract the other features for the last data point in the window
            extra_features = group.iloc[i + WINDOW_SIZE - 1][feature_names].values
            # Add the data to our processed data DataFrame
            processed_data.loc[idx] = [loc_id, timestamp, y[idx]] + list(X[idx]) + list(extra_features)
            idx += 1

    # Print the processed data
    pd.set_option('display.max_columns', None)
    print(data)
    print(processed_data)