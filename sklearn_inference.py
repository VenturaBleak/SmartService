import os
import pickle
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == '__main__':
    ############################
    # Load the best models
    ############################

    # save df to csv, save best_models to pickle
    cwd = os.getcwd()
    model_folder = os.path.join(cwd, 'models')
    ensemble_model_filename = os.path.join(model_folder, 'sklearn_ensemble_model.pickle')

    from data_preparation import prepare_data

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

    import pandas as pd
    import numpy as np

    # Load the original test data
    original_df = pd.read_csv(filename)

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

    # Concatenate the results dataframe with the original dataframe
    [original_df, results_df], axis=1)

    # Save the DataFrame to a CSV file
    merged_filename = os.path.join(model_folder, ensemble_model_filename.split('.')[0] + '_predictions.csv')
    merged_df.to_csv(merged_filename, index=False)
    print(f"Saved predictions in {merged_filename}")