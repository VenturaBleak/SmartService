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
    TARGET_COL = "kWh"
    # TARGET_COL = "Blocked_kWh"
    FORECAST_HORIZON = 4
    SCALE = True
    IMPUTE = True
    SHUFFLE = False
    TIME_WISE_SPLIT = True

    # exclude the following columns:
    EXCLUDE_COLS = ["PC6", "Date", "ChargeSocket_ID_count", "ConnectionTimeHours", "power", "effective_charging_hrs",
                    "MaxOccupancy", "SpareCap_Effective", "SpareCap_Occup_kWh", "SpareCap_Hrs", "Effective%",
                    "Occupancy_kwh%", 'WeekOfYear', 'MonthOfYear']

    print(f"loading dataset {FILE_NAME}...")
    df = pd.read_csv(FILE_NAME)
    print(f"dataset {FILE_NAME} loaded")

    # specify the forecasting columns to keep
    keep_cols = [TARGET_COL + f"(t+{FORECAST_HORIZON})",
                 "MaxPower" + f"(t+{FORECAST_HORIZON})"]

    # for the following columns keep only (t+{FORECAST_HORIZON}) and (t+0)
    additional_cols = ["number_of_chargers", "WeekIndex", "CosWeekOfYear", "SinWeekOfYear", "CosMonthOfYear", "SinMonthOfYear"]

    # create a list of columns to keep at (t+0) and (t+{FORECAST_HORIZON})
    keep_cols += [col + f"(t-0)" for col in additional_cols]
    keep_cols += [col + f"(t+{FORECAST_HORIZON})" for col in additional_cols]

    # add all columns with "(t-" that have target_col, max power, or week index in their name
    keep_cols += [col for col in df.columns if ("(t-" in col) and (TARGET_COL in col or "MaxPower" in col)]

    # set the index to the master key
    df.set_index(MASTER_KEY, inplace=True)

    # exclude all other columns with "(t+" regex or "(t-" regex that are not in keep_cols
    cols_to_drop = [col for col in df.columns if ("(t+" in col or "(t-" in col) and col not in keep_cols]
    df = df.drop(columns=cols_to_drop)

    if testing != True:
        # get index of all rows where the week index is > 45
        test_idx = df[df[f"WeekIndex(t+{FORECAST_HORIZON})"] > 45].index

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
    # scaler = MinMaxScaler(feature_range=(-1, 1))

    #  mean=0 and std=1 -> with_centering parameter to True and the with_scaling parameter to True
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(0.0, 90.0))

    if pytorch:
        # scale the data
        if SCALE:
            # scale the data
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print(f"Data scaled to mean 0 and std 1.")

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
        return X_train, X_test, y_train, y_test, X.columns, scaler, imputer