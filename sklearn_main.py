# import libraries
import math
import time
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score

# import from helper files
from sklearn_helper import adjusted_r2

def main():
    RANDOM_STATE = 42

    # save df to csv, save best_models to pickle
    cwd = os.getcwd()
    model_folder = os.path.join(cwd, 'models')

    # fetch data
    from data_preparation import prepare_data
    X_train, X_test, y_train, y_test, feature_columns = prepare_data()

    #########################################
    # Feature Importance
    #########################################
    print("----------------------------------------")
    print("Calculating feature importance...")

    # define regressor
    regressor = RandomForestRegressor(n_estimators=100,
                                      random_state=RANDOM_STATE)

    # specify pipeline
    pipeline = Pipeline([
        ("regressor", regressor)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model, using MSE and MAE as metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    # print train and test errors
    print("Regressor: Random Forest")
    print(f"Train - RMSE: {math.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}, MAE: {mean_absolute_error(y_train,  y_train_pred):.4f}")
    print(f"Test - RMSE: {math.sqrt(mean_squared_error(y_test, y_pred)):.4f}, MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    # Plot feature importances
    clf = pipeline.named_steps['regressor']
    from sklearn_helper import plot_feature_importances
    # plot top 20 features
    plot_feature_importances(clf, pd.DataFrame(X_train, columns=feature_columns), y_train, top_n=20,
                             print_table=False)
    # print all features
    feature_importance = plot_feature_importances(clf, pd.DataFrame(X_train, columns=feature_columns), y_train,
                                                  top_n=len(feature_columns), print_table=True, plot=False)
    # save feature importance to csv
    feature_importance.to_csv(os.path.join(model_folder, 'sklearn_feature_importance.csv'), index=False)

    #############################
    # Grid Search
    #############################
    # Define parameters for each regressor
    params_RF = {
        'regressor': [RandomForestRegressor(random_state=RANDOM_STATE)],
        'regressor__n_estimators': [50, 100, 500, 1000],
        'regressor__max_depth': [1, 5, 10, 20, None],
        'regressor__max_features': [1.0, 1],
        'regressor__max_leaf_nodes': [2, 9]
    }

    params_XGB = {
        'regressor': [XGBRegressor(random_state=RANDOM_STATE)],
        'regressor__n_estimators': [50, 100, 250, 1000],
        'regressor__max_depth': [1, 5, 10, 20, None],
        'regressor__learning_rate': [0.01, 0.02, 0.05, 0.1],
        'regressor__booster': ['gbtree'],
    }

    params_LGBM = {
        'regressor': [LGBMRegressor(random_state=RANDOM_STATE)],
        'regressor__n_estimators': [50, 100, 250, 1000],
        'regressor__max_depth': [-1, 1, 5, 10, 20, None],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__boosting_type': ['gbdt'],
    }

    # Put params into a list
    params_list = [params_RF, params_XGB, params_LGBM]

    best_models = {}

    print("-----------------------------")
    print("Performing grid search...")
    print("-----------------------------")

    best_parameters = {}

    # Create an empty dataframe to store the results
    results_df = pd.DataFrame(columns=['model', 'train_rmse', 'test_rmse', 'test_mae', 'r2_train', 'r2_test',
                                       'adj_r2_train', 'adj_r2_test', 'parameters'])

    # Loop over every regressor in the list
    for params in params_list:
        start_time = time.time()

        # Create a grid of parameters
        grid = ParameterGrid(params)

        best_score = float('inf')
        best_params = None
        best_model = None

        for param_set in tqdm(grid):
            pipeline.set_params(**param_set)
            pipeline.fit(X_train, y_train)

            # Predict on test set and compute error
            y_test_pred = pipeline.predict(X_test)
            score = mean_squared_error(y_test, y_test_pred)

            # If this is a better score, store these parameters
            if score < best_score:
                best_score = score
                best_params = param_set
                best_model = pipeline

        # Calculate the time taken to perform the search
        elapsed_time = time.time() - start_time

        # make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Store the best models
        best_models[best_model.named_steps['regressor'].__class__.__name__] = best_model

        # Save model parameters to a dict
        best_parameters[best_model.named_steps['regressor'].__class__.__name__] = best_params

        # Compute additional metrics
        train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        adj_r2_train = adjusted_r2(r2_train, X_train.shape[0], X_train.shape[1])
        adj_r2_test = adjusted_r2(r2_test, X_test.shape[0], X_test.shape[1])

        # Add a new row to the results dataframe
        new_row = {'model': best_model.named_steps['regressor'].__class__.__name__,
                   'train_rmse': train_rmse,
                   'test_rmse': test_rmse,
                   'test_mae': test_mae,
                   'r2_train': r2_train,
                   'r2_test': r2_test,
                   'adj_r2_train': adj_r2_train,
                   'adj_r2_test': adj_r2_test,
                   'parameters': best_params}

        # append the new row to the dataframe
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # print model performance
        print(f"Best model performance: "
              f"Train RMSE: {math.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}, "
              f"Test RMSE: {math.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}, "
              f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}, "
              f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Best model parameters: {best_parameters[best_model.named_steps['regressor'].__class__.__name__]}")
        print("-------------------------------------------")

    # sort the dataframe by test_rmse
    results_df = results_df.sort_values(by='test_rmse')

    # save the results to a csv file
    results_df.to_csv(os.path.join(model_folder, 'sklearn_best_models_results.csv'), index=False)

    ############################
    # Compare the best models
    ############################
    import matplotlib as mpl
    # Increase the size of all text in the plot
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16

    # Create the main axis for RMSE
    ax1 = plt.gca()
    ax1.grid(False)

    # Create a twin axis for MAE
    ax2 = ax1.twinx()

    # Plot RMSE values
    rects1 = ax1.bar(results_df['model'], results_df['test_rmse'], label='Test RMSE', alpha=0.8)
    ax1.bar_label(rects1, labels=results_df['test_rmse'].round(2), label_type='edge')
    ax1.set_ylabel('Test RMSE')
    ax1.set_ylim(results_df['test_rmse'].min() * 0.98, results_df['test_rmse'].max() * 1.02)

    # Plot MAE values
    rects2 = ax2.bar(results_df['model'], results_df['test_mae'], label='Test MAE', alpha=0.3)
    ax2.bar_label(rects2, labels=results_df['test_mae'].round(2), label_type='edge')
    ax2.set_ylabel('Test MAE')

    # Set other properties
    plt.title('Comparison of Test RMSE and Test MAE for the Best Regressors')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show the legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    # Show the plot
    plt.show()

    ############################
    # Save the best models
    ############################

    # After the grid search loop, concatenate the training and testing sets
    print("X_train type:", type(X_train)) # is a numpy array
    print("X_test type:", type(X_test)) # is a numpy array
    print("y_train type:", type(y_train)) # is a pandas series
    print("y_test type:", type(y_test)) # is a pandas series

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # del X_train, X_test, y_train, y_test
    del X_train, X_test, y_train, y_test

    # load inference data
    print("\nLoading inference data...")
    from data_preparation import prepare_data
    X_test, _, y_test, _, _, filename = prepare_data(testing=True)

    # Create an Ensemble model of best models
    print("\nCreating an Ensemble model of best models...")
    models_tuples = [(name, model) for name, model in best_models.items()]
    ensemble = VotingRegressor(estimators=models_tuples)
    ensemble.fit(X, y)

    # Perform inference on the test set
    print("\nPerforming inference on the test set...")
    y_test_pred = ensemble.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"Ensemble Model Test RMSE: {test_rmse:.4f}")
    print(f"Ensemble Model Test MAE: {test_mae:.4f}")

    # Save the Ensemble model to a pickle file
    print("\nSaving the Ensemble model to a pickle file...")
    ensemble_model_filename = os.path.join(model_folder, 'sklearn_ensemble_model.pickle')
    with open(ensemble_model_filename, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"Ensemble model saved as {ensemble_model_filename}")

if __name__ == "__main__":
    main()