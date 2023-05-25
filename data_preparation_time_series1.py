import pandas as pd
import numpy as np

# Function to create sliding windows for the time series data
def create_sliding_window(data, observation_period, forecasting_period):
    X, y = [], []
    # Loop over the data to create sliding windows
    for i in range(len(data) - observation_period - forecasting_period + 1):
        # Append the windowed data to X (input features)
        # and the next data point to y (output)
        # Note that we reverse the windowed data so it's in order t, t-1, t-2, ...
        X.append(data[i:i + observation_period][::-1])
        y.append(data[i + observation_period:i + observation_period + forecasting_period])
    return np.array(X), np.array(y)

# Function to prepare the time series data
def prepare_time_series_data(data, observation_period, forecasting_period, location_column, time_column, target_column):
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
        X_group, y_group = create_sliding_window(kwh_data, observation_period, forecasting_period)
        # Add the windows to our lists
        X.extend(X_group)
        y.extend(y_group)
    return np.array(X), np.array(y), grouped_data.groups.keys(), grouped_data

if __name__ == '__main__':
    TRY = True

    if TRY == True:
        filename = "test.csv"

        # Example usage
        data_df = pd.DataFrame({
            'PC6': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'Date': pd.date_range('2023-01-01', periods=6).tolist() * 2,
            'kwh': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155, 200, 210],
            'blocked_kwh': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155, 200, 210],
            'MaxPower': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155, 200, 210],
            'Population': [5000, 5100, 5150, 5200, 5250, 3000, 3100, 3200, 3300, 3350, 200, 210],
            'Avg Income': [60000, 60500, 61000, 61200, 61500, 80000, 80500, 81000, 81500, 82000, 200, 210],
            'Avg Household Size': [6, 6, 6, 7, 6, 8, 7, 8, 8, 9, 10, 10]
        })\

        # Define parameters for the time series data
        TARGET_COLUMNS = ['kwh', 'MaxPower', 'blocked_kwh']
        LOCATION_COLUMN = 'PC6'
        TIME_COLUMN = 'Date'
        OBSERVATION_PERIOD = 3
        FORECASTING_PERIOD = 3

    else:
        # load csv file named processed_data.csv
        filename = 'final_data_SE_cleaned.csv'
        data_df = pd.read_csv(filename)

        # Define parameters for the time series data
        TARGET_COLUMNS = ['MaxPower', 'kWh', 'Blocked_kWh', 'WeekIndex']
        LOCATION_COLUMN = 'PC6'
        TIME_COLUMN = 'Date'
        OBSERVATION_PERIOD = 4
        FORECASTING_PERIOD = 4

    # initialize pd df
    processed_data = pd.DataFrame()

    # Create a dictionary to store the processed data for each target column
    processed_data_dict = {}

    for counter, TARGET_COLUMN in enumerate(TARGET_COLUMNS):
        print(f'Processing {TARGET_COLUMN}...')
        data = data_df.copy()

        if counter +1 != len(TARGET_COLUMNS):
            # drop every column except the target column, the time column and the location column
            data = data.drop([col for col in data.columns if col not in  [TARGET_COLUMN, TIME_COLUMN, LOCATION_COLUMN]], axis=1)


        if counter +1 == len(TARGET_COLUMNS):
            if TRY != True:
                # free up memory
                del data_df

        # Prepare the data
        X, y, loc_ids, grouped_data = prepare_time_series_data(data, OBSERVATION_PERIOD, FORECASTING_PERIOD,
                                                               LOCATION_COLUMN,
                                                               TIME_COLUMN, TARGET_COLUMN)
        print('time series data prepared')

        # fetch feature names, that is all columns except the target column and the time column and the location column
        feature_names = [col for col in data.columns if col not in TARGET_COLUMNS + [TIME_COLUMN, LOCATION_COLUMN]]

        # free up memory
        del data

        # Generate dynamic column names based on the window size
        column_names = [
            [LOCATION_COLUMN, TIME_COLUMN] + [f'{TARGET_COLUMN}(t+{i})' for i in range(FORECASTING_PERIOD, 0, -1)] + [
                f'{TARGET_COLUMN}(t-{i})' for i in range(OBSERVATION_PERIOD)] + feature_names]

        # Initialize an empty DataFrame with the dynamic column names
        processed_data_temp = pd.DataFrame(columns=column_names)

        idx = 0
        processed_data_list = []
        # Loop over each group
        for loc_id in loc_ids:
            group = grouped_data.get_group(loc_id)
            timestamps = group.iloc[OBSERVATION_PERIOD - 1:-FORECASTING_PERIOD][TIME_COLUMN].values
            extra_features = group.iloc[OBSERVATION_PERIOD - 1:-FORECASTING_PERIOD][feature_names].values
            group_data = [[loc_id, timestamps[i]] + list(y[i][::-1]) + list(X[i]) + list(extra_features[i])
                          for i in range(len(timestamps))]
            processed_data_list.extend(group_data)

        # Initialize a DataFrame with the data list
        processed_data_temp = pd.DataFrame(processed_data_list, columns=column_names)

        # Add the processed data to the dictionary
        processed_data_dict[TARGET_COLUMN] = processed_data_temp

    # Print the processed data
    pd.set_option('display.max_columns', None)

    # merge the dataframes that are stored in the dictionary
    processed_data = pd.concat(processed_data_dict.values(), axis=1)

    # remove duplicate columns
    processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]

    if TRY == True:
        print(data_df)
        print(processed_data)

    # save to new csv file named original name + _cleaned
    processed_data.to_csv(filename.split('.')[0] + '_processed.csv', index=False)