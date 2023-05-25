import pandas as pd
import numpy as np
from tqdm import tqdm

# Function to create sliding windows for the time series data
def create_sliding_window(data, observation_period, forecasting_period):
    observation_np, forecast_np = [], []
    # Loop over the data to create sliding windows
    for i in range(len(data) - observation_period - forecasting_period + 1):
        # Append the windowed data to X (input features)
        # and the next data point to y (output)
        # Note that we reverse the windowed data so it's in order t, t-1, t-2, ...
        observation_np.append(data[i:i + observation_period][::-1])
        forecast_np.append(data[i + observation_period:i + observation_period + forecasting_period])
    return np.array(observation_np), np.array(forecast_np)

# Function to prepare the time series data
def prepare_time_series_data(data, observation_period, forecasting_period, location_column, time_column, target_column):
    # Sort the data by Location ID and Timestamp
    data = data.sort_values([location_column, time_column])
    # Group the data by Location ID
    grouped_data = data.groupby(location_column)

    observation_values, forecast_values = [], []
    # Loop over each group
    for _, group in grouped_data:
        # Extract the target time series data
        kwh_data = group[target_column].values
        # Create sliding windows for this group
        observation_group, forecast_group = create_sliding_window(kwh_data, observation_period, forecasting_period)
        # Add the windows to our lists
        observation_values.extend(observation_group)
        forecast_values.extend(forecast_group)

    return np.array(observation_values), np.array(forecast_values), grouped_data.groups.keys(), grouped_data

if __name__ == '__main__':
    TRY = False

    if TRY == True:
        filename = "test.csv"

        # Example usage
        # data_df = pd.DataFrame({
        #     'PC6': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        #     'Date': pd.date_range('2023-01-01', periods=6).tolist() * 2,
        #     'kwh': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155, 200, 210],
        #     'blocked_kwh': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155, 200, 210],
        #     'MaxPower': [100, 110, 95, 105, 120, 150, 140, 160, 130, 155, 200, 210],
        #     'Population': [5000, 5100, 5150, 5200, 5250, 3000, 3100, 3200, 3300, 3350, 200, 210],
        #     'Avg Income': [60000, 60500, 61000, 61200, 61500, 80000, 80500, 81000, 81500, 82000, 200, 210],
        #     'Avg Household Size': [6, 6, 6, 7, 6, 8, 7, 8, 8, 9, 10, 10]
        # })\

        # save the data to csv
        # data_df.to_csv(filename, index=False)

        # load csv file named processed_data.csv
        data_df = pd.read_csv(filename)

        # Define parameters for the time series data
        TARGET_COLUMNS = ['kwh', 'MaxPower', 'blocked_kwh']
        LOCATION_COLUMN = 'PC6'
        TIME_COLUMN = 'Date'
        OBSERVATION_PERIOD = 3
        FORECASTING_PERIOD = 3

    else:
        # load csv file named processed_data.csv
        filename = 'final_data_SE_cleaned.csv'
        print(f"loading dataset {filename}...")
        data_df = pd.read_csv(filename)
        print(f"dataset {filename} loaded")

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
        observation_arrays, forecast_arrays, loc_ids, grouped_data = prepare_time_series_data(data, OBSERVATION_PERIOD, FORECASTING_PERIOD,
                                                               LOCATION_COLUMN,
                                                               TIME_COLUMN, TARGET_COLUMN)

        # print(f"X: {X}, y: {y}, loc_ids: {loc_ids}, grouped_data: {grouped_data}")
        # print('time series data prepared')

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

        print(f"looping over loc_ids: {len(loc_ids)}")
        # In the loop over each group
        data_rows = []  # List to hold the data rows


        if counter == 0:
            print(f"obtaining row count...")
            row_count = 0
            # count once to instantiate list with predefined number of rows
            for loc_id in tqdm(loc_ids, desc="Processing Location IDs"):
                group = grouped_data.get_group(loc_id)
                timestamps = group.iloc[OBSERVATION_PERIOD - 1:-FORECASTING_PERIOD][TIME_COLUMN].values

                # Loop over the remaining timestamps in the group
                for i in range(len(timestamps)):
                    row_count +=1

            print(f"row count: {row_count}")


        # instantiate empty list of length = row count in order to avoid appending to list, for performance reasons
        data_rows = [None] * row_count

        counts = 0
        # count once to instantiate list with predefined number of rows
        for loc_id in tqdm(loc_ids, desc="Processing Location IDs"):
            group = grouped_data.get_group(loc_id)
            timestamps = group.iloc[OBSERVATION_PERIOD - 1:-FORECASTING_PERIOD][TIME_COLUMN].values
            extra_features = group.iloc[OBSERVATION_PERIOD - 1:-FORECASTING_PERIOD][feature_names].values
            print(f"extra features:\n {extra_features}")

            # Loop over the remaining timestamps in the group
            for i in range(len(timestamps)):
                try:
                    row = dict(zip(column_names[0], [loc_id, timestamps[i]] + list(forecast_arrays[counts][::-1]) + list(
                        observation_arrays[counts]) + list(extra_features[i])))
                    print(f"extra features [i]: {extra_features[i]}")
                except:
                    row = dict(zip(column_names[0], [loc_id, timestamps[i]] + list(forecast_arrays[counts][::-1]) + list(
                        observation_arrays[counts]) + list(extra_features[0])))
                    print(f"extra features [0]: {extra_features[0]}")
                data_rows[counts] = row
                counts +=1

        # Create DataFrame from list of dictionaries
        processed_data_temp = pd.DataFrame(data_rows)
        del data_rows, loc_ids, grouped_data, observation_arrays, forecast_arrays, extra_features

        # Store the processed data for this target column
        processed_data_dict[TARGET_COLUMN] = processed_data_temp
        del processed_data_temp

    # Print the processed data
    pd.set_option('display.max_columns', None)

    # merge the dataframes that are stored in the dictionary
    processed_data = pd.concat(processed_data_dict.values(), axis=1)
    del processed_data_dict

    # remove duplicate columns
    processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]

    if TRY == True:
        print(data_df)
        print(processed_data)

    print(f"saving processed data...")
    # save to new csv file named original name + _cleaned
    processed_data.to_csv(filename.split('.')[0] + '_processed.csv', index=False)
    print(f"Sucess! processed data saved to {filename.split('.')[0] + '_processed.csv'}")