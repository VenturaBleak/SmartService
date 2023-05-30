"""
Script: data_preparation_time_series.py
========================
This script serves as a utility for preparing time series data for machine learning models. It follows a sliding window approach to form observation and forecast sets from past and future values respectively.

The `create_sliding_window` function creates windows from the time series data. The `prepare_time_series_data` function handles the overall preparation process, which involves sorting, grouping, and windowing the data.

The core parameters for this script include:
    - LOCATION_COLUMN: Column to group the data by.
    - TIME_COLUMN: Time-based column for sorting.
    - TARGET_COLUMNS: Columns for the target variable.
    - OBSERVATION_PERIOD: Length of the observation window.
    - FORECASTING_PERIOD: Length of the forecast window.

The script outputs a DataFrame, where each row represents a sliding window for a specific group. This DataFrame is saved as a CSV file, facilitating the subsequent training and testing stages.

A utility for splitting the prepared data into training and testing datasets is also provided, based on a specific date. This allows for easy model validation.

For a trial run, the TRY variable can be set to True. In this case, a test dataset is used to provide a quick overview of the process.

In summary, this script provides a comprehensive solution for preparing time series data for machine learning models, adhering to best practices and guidelines.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Function to create sliding windows for the time series data
def create_sliding_window(data, observation_period, forecasting_period):
    """
    Creates sliding windows of given lengths for the provided time series data.

    Args:
        data (np.array): The time series data.
        observation_period (int): The length of the observation window.
        forecasting_period (int): The length of the forecasting window.

    Returns:
        tuple: Two numpy arrays containing the observation and forecast windows.
    """
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
    """
       Prepares the time series data by creating sliding windows for each group of the data.

       Args:
           data (pd.DataFrame): The DataFrame containing the data.
           observation_period (int): The length of the observation window.
           forecasting_period (int): The length of the forecasting window.
           location_column (str): The name of the column containing the location data.
           time_column (str): The name of the column containing the time data.
           target_column (str): The name of the column containing the target data.

       Returns:
           tuple: Two numpy arrays containing the observation and forecast windows for each group,
                  a list of the groups' keys, and a GroupBy object of the data grouped by location.
    """

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
        OBSERVATION_PERIOD = 6
        FORECASTING_PERIOD = 3

    else:
        # load csv file named processed_data.csv
        filename = 'final_data_cleaned.csv'
        print(f"loading dataset {filename}...")
        data_df = pd.read_csv(filename, low_memory=False)
        print(f"dataset {filename} loaded")

        # Define parameters for the time series data
        # open pickled dictionary
        import pickle
        with open('final_data_column_lists.pickle', 'rb') as handle:
            column_lists = pickle.load(handle)

        print(f"column_lists: {column_lists}")

        # drop descriptive columns
        data_df = data_df.drop([col for col in data_df.columns if col in column_lists['descriptive_columns']], axis=1)

        TARGET_COLUMNS = column_lists['target_columns'] + column_lists['duplicate_target_columns'] + column_lists['lagged_columns']
        LOCATION_COLUMN = column_lists['identifier_columns'][0]
        TIME_COLUMN = column_lists['identifier_columns'][1]
        OBSERVATION_PERIOD = 8
        FORECASTING_PERIOD = 4

        print(f"TARGET_COLUMNS: {TARGET_COLUMNS}")
        print(f"LOCATION_COLUMN: {LOCATION_COLUMN}")
        print(f"TIME_COLUMN: {TIME_COLUMN}")

    # initialize pd df
    processed_data = pd.DataFrame()

    # Create a dictionary to store the processed data for each target column
    processed_data_dict = {}

    for counter, TARGET_COLUMN in enumerate(TARGET_COLUMNS):
        print(f'Processing {TARGET_COLUMN}...')
        data = data_df.copy()
        # print(f"data: {data}")

        if counter +1 != len(TARGET_COLUMNS):
            # drop every column except the target column, the time column and the location column
            data = data.drop([col for col in data.columns if col not in [TARGET_COLUMN, TIME_COLUMN, LOCATION_COLUMN]], axis=1)

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
            # print(f"extra features:\n {extra_features}")

            # Loop over the remaining timestamps in the group
            for i in range(len(timestamps)):
                try:
                    row = dict(zip(column_names[0], [loc_id, timestamps[i]] + list(forecast_arrays[counts][::-1]) + list(
                        observation_arrays[counts]) + list(extra_features[i])))
                    # print(f"extra features [i]: {extra_features[i]}")
                except:
                    row = dict(zip(column_names[0], [loc_id, timestamps[i]] + list(forecast_arrays[counts][::-1]) + list(
                        observation_arrays[counts]) + list(extra_features[0])))
                    # print(f"extra features [0]: {extra_features[0]}")
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

    # create an empty list to hold the column names to drop
    cols_to_drop = []

    # loop over columns in column_lists['target_columns'] and column_lists['duplicate_target_columns']
    for list_name in ['duplicate_target_columns']:
        for col in column_lists[list_name]:
            # concatenate string: column name +(t+
            col_t_plus = col + '(t+'
            # add all columns whose names include: col_t_plus to cols_to_drop
            cols_to_drop += [c for c in processed_data.columns if col_t_plus in c]

    # loop over columns in column_lists['lagged_columns']
    for col in column_lists['lagged_columns']:
        col_t_plus = col + '(t-'
        col_t0 = col + '(t-0)'
        # add all columns whose names include: col_t_plus except if the name also includes col_t0
        cols_to_drop += [c for c in processed_data.columns if col_t_plus in c and col_t0 not in c]

    # remove duplicate column names
    cols_to_drop = list(set(cols_to_drop))

    # drop the columns
    processed_data = processed_data.drop(columns=cols_to_drop)

    # get latest date of the date column and drop all rows with date >= latest date
    latest_date = processed_data['Date'].max()

    # drop all rows with date >= latest date
    processed_data = processed_data[processed_data['Date'] < latest_date]

    print(f"saving processed data...")

    # save to new csv file named original name + _cleaned
    processed_data.to_csv(filename[:-4] + '_processed.csv', index=False)

    # save two files, one for training and one for testing; keep as everything with date >= 01-01-2023 for testing, rest for training
    processed_data[processed_data['Date'] < '2023-01-01'].to_csv(filename[:-4] + '_processed_train.csv', index=False)
    processed_data[processed_data['Date'] >= '2023-01-01'].to_csv(filename[:-4] + '_processed_test.csv', index=False)

    print(f"Success! Processed data saved to {filename.split('.')[0] + '_processed.csv'}")