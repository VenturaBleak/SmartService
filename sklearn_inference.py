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
    ensemble_model_filename = os.path.join(model_folder, 'ensemble_model.pickle')

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