"""
This Python script is responsible for preparing data for machine learning analysis and model training. It's primarily used within Google's Data Science pipelines.

Functionality:
The script includes importing required packages, defining the function 'prepare_data' with options for PyTorch compatibility and testing mode, specifying various parameters and columns for processing, and loading a dataset from a CSV file.

The script processes the dataset by setting a specific column as the index, applying specified transformations and dropping unnecessary columns, splitting the data into training and testing sets with an option for time-series split, and assuring no data leakage between these sets.

When the PyTorch mode is enabled, the script imputes missing values using KNNImputer. If the script is not in testing mode, it prints out information about the final datasets, including feature names, numbers of features and samples.

Usage:
The 'prepare_data' function can be called with two optional boolean parameters - 'pytorch' and 'testing'. By default, 'pytorch' is set to True and 'testing' is set to False. When in testing mode, the function will load a testing dataset and minimize console output. When PyTorch compatibility is enabled, the function will return imputed datasets ready for PyTorch-based machine learning models.

Module Dependencies:
- pandas: Required for data handling and manipulation.
- sklearn.model_selection: Used for splitting the dataset into training and testing sets.
- sklearn.impute: Contains the KNNImputer class for handling missing data.
- sklearn.preprocessing: Includes Scaler classes for data normalization.
- pickle: Used for loading a pickled dictionary that contains lists of columns to process in the dataset.

Note: To execute this script correctly, ensure the required CSV files and a 'final_data_column_lists.pickle' file are available in the same directory as the script.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def prepare_data(pytorch=True, testing=False):
    # specify coluns and parameters
    RANDOM_STATE = 42
    FILE_NAME = "final_data_cleaned_processed"
    if testing == True:
        FILE_NAME = FILE_NAME + "_test" + ".csv"
    else:
        FILE_NAME = FILE_NAME + "_train" + ".csv"
    MASTER_KEY = "PC6_WeekIndex"
    TARGET_COL = "Consumed_kWh"
    # TARGET_COL = "Blocked_kWh"
    FORECAST_HORIZON = 4
    IMPUTE = True
    SHUFFLE = False
    TIME_WISE_SPLIT = True

    # exclude the following columns:
    EXCLUDE_COLS = ["PC6", "Date"]

    print(f"loading dataset {FILE_NAME}...")
    df = pd.read_csv(FILE_NAME)

    # set the index to the master key
    df.set_index(MASTER_KEY, inplace=True)

    print(f"dataset {FILE_NAME} loaded")

    # open pickled dictionary
    import pickle
    with open('final_data_column_lists.pickle', 'rb') as handle:
        column_lists = pickle.load(handle)

    # specify the forecasting columns to keep
    keep_cols = [TARGET_COL + f"(t+{FORECAST_HORIZON})",
                 column_lists['lagged_columns'][0] + f"(t+{FORECAST_HORIZON})",
                 column_lists['lagged_columns'][1] + f"(t+{FORECAST_HORIZON})"]

    # Assuming you already have a list of columns to keep
    cols_to_drop = []

    # loop over columns in column_lists['target_column']
    for col in column_lists['target_columns']:
        if col == TARGET_COL:
            print()
            # if col + (t- is in the column name, add it to keep_cols
            cols_to_drop += [c for c in df.columns if col + '(t-' in c]
        else:
            # drop all other columns with col in the name
            cols_to_drop += [c for c in df.columns if col in c]

    # drop all other columns with "(t+" regex
    cols_to_drop += [col for col in df.columns if "(t+" in col and col not in keep_cols]

    # remove duplicates from cols_to_drop
    cols_to_drop = list(set(cols_to_drop))

    # sort by column name
    cols_to_drop.sort()

    print(f"dropping {len(cols_to_drop)} columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    if testing != True:
        # get index of all rows where the week index is > 45
        test_idx = df[df["Date"] > '2022-11-01'].index

    # split in X and y
    y = df[TARGET_COL+f"(t+{FORECAST_HORIZON})"]

    # drop the target column from the X, as well as the columns in EXCLUDE_COLS
    if EXCLUDE_COLS[0] is not None:
        EXCLUDE_COLS = EXCLUDE_COLS + [TARGET_COL+f"(t+{FORECAST_HORIZON})"]
    else:
        EXCLUDE_COLS = [TARGET_COL+f"(t+{FORECAST_HORIZON})"]
    X = df.drop(EXCLUDE_COLS, axis=1)
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")

    if testing == True:
        # split in train and test, test only as placeholder
        X_train = X
        X_test = X.iloc[:1, :]
        y_train = y
        y_test = y.iloc[:1]

    else:
        if TIME_WISE_SPLIT == True:
            # train != test index, test == test index
            X_train = X[~X.index.isin(test_idx)]
            X_test = X[X.index.isin(test_idx)]
            y_train = y[~y.index.isin(test_idx)]
            y_test = y[y.index.isin(test_idx)]
            print(f"y_train {y_train[1:100]} y_test {y_test[1:100]}")
        else:
            # split in train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=SHUFFLE, stratify=None,
                                                                random_state=RANDOM_STATE)

    # Unit Test - data leakage: Check the indices of the train and test data after the train_test_split function to ensure that there is no overlap between them
    assert len(
        set(X_train.index).intersection(
            set(X_test.index))) == 0, "Data leakage detected: Train and test sets have common ids."

    print(f"X type: {type(X_test)}, X.shape {X_test.shape}, y type: {type(y_test)}, y.shape {y_test.shape}")

    # print dataset information
    print(f"Feature names: {X.columns.to_list()}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of testing samples: {X_test.shape[0]}")

    # impute missing values
    imputer = KNNImputer(n_neighbors=5)

    if pytorch:
        if IMPUTE:
            # impute missing values
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            # print nubmer of imputed values
            print("Number of imputed values in training set: ", pd.DataFrame(X_train).isna().sum().sum())
            print("Number of imputed values in testing set: ", pd.DataFrame(X_test).isna().sum().sum())
        if testing == True:
            # return the data
            return X_train, X_test, y_train, y_test, X.columns, FILE_NAME
        else:
            return X_train, X_test, y_train, y_test, X.columns

    else:
        # return the data, scaler, and imputer
        return X_train, X_test, y_train, y_test, X.columns, imputer