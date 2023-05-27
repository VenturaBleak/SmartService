import torch
import os
import random
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset import CustomDataset
from train import eval_fn
from utils import load_checkpoint

def main():
    # Set the model parameters
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    if DEVICE == "cuda":
        NUM_WORKERS = 2
    else:
        NUM_WORKERS = 0
    PIN_MEMORY = True

    # save the merged dataframe -> split the .csv from filename and add _predictions.csv
    cwd = os.getcwd()

    ############################
    # set seeds
    ############################
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = False

    ###################################################################################################################
    #### Specify Training, and Test Datasets
    ###################################################################################################################
    # fetch data
    from data_preparation import prepare_data
    X_test, _, y_test, _, _, filename = prepare_data(testing=True)

    # convert y_train and y_test to numpy arrays
    y_test = y_test.to_numpy()

    # specify input and output sizes
    INPUT_SIZE = X_test.shape[-1]
    OUTPUT_SIZE = 1

    ############################
    # Create data loaders
    ############################
    # create testing data loader
    test_dataset = CustomDataset(X_test, y_test)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    ############################
    # Load model
    ############################
    # FC model
    from model import MLP
    NUM_HIDDEN_LAYERS = 8
    NODES_PER_LAYER = 400
    MODEL_NAME = "best_model"

    model = MLP(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER,
        dropout_rate=0.05
    )
    model.to(DEVICE)
    model, _ = load_checkpoint(MODEL_NAME, model)

    ############################
    # Loss function
    ############################
    loss_fn = torch.nn.MSELoss(reduction='mean')

    ############################
    # Evaluate model
    ############################

    test_loss, mse, rmse, mae = eval_fn(test_data_loader, model, loss_fn, DEVICE)

    print(f"Test Loss: {test_loss:.4f}, Test MSE: {mse:.2f}, Test RMSE: {rmse:.2f}, Test MAE: {mae:.2f}")

    # save the results in a csv file
    results_dict = {
        'test_loss': [test_loss],
        'test_mse': [mse],
        'test_rmse': [rmse],
        'test_mae': [mae]
    }
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(cwd, 'models', filename.split('.')[0] + '_inference_results.csv'), index=False)

    # Generate predictions
    model.eval()
    y_pred = []
    diffs = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_data_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            output = model(X_batch)
            y_pred_batch = output.cpu().numpy().flatten()
            y_pred.extend(y_pred_batch)

            # Store actuals, as floats
            actuals.extend(y_batch.cpu().numpy().flatten().tolist())

        # actuals - predictions
        diffs = np.array(actuals) - np.array(y_pred)

    # populate validation results dictionary
    valid_results_dict = {
        'actuals': actuals,
        'predictions': y_pred,
        'diffs': diffs
    }
    del actuals, y_pred, diffs

    valid_res = pd.DataFrame(valid_results_dict)
    del valid_results_dict

    # Make sure that all values are float type.
    valid_res['predictions'] = valid_res['predictions'].astype(float)
    valid_res['actuals'] = valid_res['actuals'].astype(float)
    valid_res['diffs'] = valid_res['diffs'].astype(float)

    # Get the absolute value of the differences for size of scatterplot dots
    valid_res['abs_diffs'] = valid_res['diffs'].abs()

    # Calculate min and max values across both predictions and actuals to unify axes
    max_val = np.max([valid_res['predictions'].max(), valid_res['actuals'].max()])
    min_val = np.min([valid_res['predictions'].min(), valid_res['actuals'].min()])

    # Plot the scatterplot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=valid_res, x='predictions', y='actuals', hue='diffs',
                    palette='coolwarm', legend='brief', sizes=(20, 200))

    # Add a 45 degree line
    plt.plot([min_val, max_val], [min_val, max_val], 'm--')

    # Set the axes limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Add labels
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')

    plt.title('Actuals vs Predictions')
    plt.grid(True)

    plt.show()
    # save figure
    plt.savefig(os.path.join(cwd, 'models', filename.split('.')[0] + '_predictions.png'))
    plt.close()

    # read original csv file =filename
    original_df = pd.read_csv(filename, index_col=0)

    # index of valid res = index=original_df.index
    valid_res = valid_res.set_index(original_df.index)

    # concatenate the valid_res dataframe with the original dataframe
    merged_df = pd.concat([original_df, valid_res], axis=1)
    del original_df, valid_res

    merged_df.to_csv(os.path.join(cwd, 'models', filename.split('.')[0] + '_predictions.csv'))
    print(f"Saved predictions in {os.path.join(cwd, 'models', filename.split('.')[0] + '_predictions.csv')}")

if __name__ == '__main__':
    main()