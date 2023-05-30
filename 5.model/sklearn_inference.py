"""
Script: sklearn_inference.py
============================
Performs inference on test data using a trained sklearn ensemble model.

This script reads the sklearn ensemble model from a pickle file, performs inference on the test set, and saves the
prediction results to a csv file.

Steps:
------
1. Load the ensemble model from a pickle file.
2. Perform inference on the test set.
3. Calculate metrics (RMSE and MAE) for the model's performance on the test set.
4. Load the original test data.
5. Prepare the prediction results, including actuals, predictions, and the differences between them.
6. Merge these results with the original test data.
7. Save the merged data to a csv file.
"""

import os
import pickle
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn_helper import adjusted_r2

if __name__ == '__main__':
    ############################
    # Load the best models
    ############################

    # save df to csv, save best_models to pickle
    cwd = os.getcwd()
    model_folder = os.path.join(cwd, 'models')
    ensemble_model_filename = os.path.join(model_folder, 'sklearn_ensemble_model.pickle')

    from data_feeding import prepare_data

    X_test, _, y_test, _, _, filename = prepare_data(testing=True)

    # load model from pickle file
    print("\nLoading the Ensemble model from a pickle file...")
    with open(ensemble_model_filename, 'rb') as f:
        ensemble_model = pickle.load(f)

    # Perform inference on the test set
    print("\nPerforming inference on the test set...")
    y_test_pred = ensemble_model.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"Ensemble Model Test RMSE: {test_rmse:.4f}")
    print(f"Ensemble Model Test MAE: {test_mae:.4f}")

    # New metric calculations, placed after the existing ones
    r2_test = r2_score(y_test, y_test_pred)
    adj_r2_test = adjusted_r2(r2_test, X_test.shape[0], X_test.shape[1])

    # DataFrame for result storage
    # This replaces the existing 'results_dict' definition
    results_df = pd.DataFrame({
        'model': ['ensemble_model'],
        'test_rmse': [test_rmse],
        'test_mae': [test_mae],
        'r2_test': [r2_test],
        'adj_r2_test': [adj_r2_test]
    })

    # Save DataFrame to CSV, placed at the end of the script before saving original_df
    results_df.to_csv(os.path.join(model_folder, ensemble_model_filename.split('.')[0] + '_inference_results.csv'),
                      index=False)
    print(f"Saved inference results in {filename}")

    # Load the original test data
    original_df = pd.read_csv(filename)

    # drop rows where "PC6" == 1059CM or 1018VN (outliers in target variable)
    outlier_PC6 = ["1059CM", "1018VN"]
    original_df = original_df[~original_df['PC6'].isin(outlier_PC6)]

    # Prepare the prediction results
    results_dict = {
        'actuals': y_test,
        'predictions': y_test_pred,
        'diffs': np.abs(np.array(y_test) - np.array(y_test_pred))
    }

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results_dict)

    # Make sure that all values are float type
    results_df['predictions'] = results_df['predictions'].astype(float)
    results_df['actuals'] = results_df['actuals'].astype(float)
    results_df['diffs'] = results_df['diffs'].astype(float)

    # reset the index for results_df
    results_df.reset_index(drop=True, inplace=True)

    # make sure the index is same for both dataframes before merging
    original_df.reset_index(drop=True, inplace=True)

    # Add new columns to original dataframe
    original_df = original_df.assign(predictions=results_df['predictions'],
                                     actuals=results_df['actuals'],
                                     diffs=results_df['diffs'])

    # Save the DataFrame to a CSV file
    filename = os.path.join(model_folder, ensemble_model_filename.split('.')[0] + '_predictions.csv')
    original_df.to_csv(filename, index=False)
    print(f"Saved predictions in {filename}")