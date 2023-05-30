"""
Script: main.py
================
This script should be used as the main entry point for training the model and should be run after the data has been properly prepared and the model and training modules have been correctly set up.

The script follows these steps:
1. Sets up a list of hyperparameters for the model, such as learning rate, device, batch size, and number of epochs. These hyperparameters can be adjusted depending on the specific use-case.
2. Initializes the random seed for reproducibility across multiple runs.
3. Retrieves the training and testing datasets using the custom prepare_data function, and creates PyTorch DataLoader objects for them using the CustomDataset class.
4. Creates a model instance.
5. Initializes a loss function, optimizer, and learning rate scheduler. The default loss function is SmoothL1Loss (Huber Loss), which is robust to outliers. The optimizer is AdamW, which includes weight decay. The learning rate scheduler is Polynomial Decay, and Gradual Warmup Scheduler.
6. Print a summary of the model using the torchinfo package.
7. train_fn and eval_fn functions from the training module to train the model and evaluate it on the testing dataset. The training process is logged with a progress bar that displays the training loss for each epoch. It also logs the testing loss, Root Mean Square Error (RMSE), and Mean Absolute Error (MAE) every 5 epochs.
8. If the testing loss in the current epoch is lower than in all previous epochs, the script saves the model's state, including the model parameters and optimizer state, to a file.
"""

import torch
import pandas as pd
import os
import random
import numpy as np

def main():
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 42
    LEARNING_RATE = 2e-4 # (0.0001)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 2048
    NUM_EPOCHS = 50
    if DEVICE == "cuda":
        NUM_WORKERS = 2
    else:
        NUM_WORKERS = 0
    PIN_MEMORY = True
    WARMUP_EPOCHS = int(NUM_EPOCHS * 0.05) # 5% of the total epochs
    # set FULL_DS to True if you want to train on the full dataset, else the train ds will be split into train and test
    FULL_DS = False

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
    X_train, X_test, y_train, y_test, _ = prepare_data()

    if FULL_DS == True:
        # concatenate X_train and X_test
        X_train = np.concatenate([X_train, X_test])
        y_train = pd.concat([y_train, y_test])

        # fetch data
        X_test, _, y_test, _, _, _ = prepare_data(testing=True)

    # convert y_train and y_test to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # specify input and output sizes
    INPUT_SIZE = X_train.shape[-1]
    OUTPUT_SIZE = 1

    ############################
    # create data loaders
    ############################
    # import data loader
    from torch.utils.data import DataLoader
    from dataset import CustomDataset

    # create training data loader
    train_dataset = CustomDataset(X_train, y_train)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    # create testing data loader
    test_dataset = CustomDataset(X_test, y_test)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Unit test the data class and data loader
    from utils import test_tensor_shapes
    test_tensor_shapes(train_data_loader, test_data_loader, INPUT_SIZE)

    ############################
    # create model
    ############################
    # FC model
    from model import MLP
    NUM_HIDDEN_LAYERS = 8
    NODES_PER_LAYER = 100

    model = MLP(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER,
        dropout_rate=0.05
    )
    model.to(DEVICE)

    # MLP with BatchNorm
    # from model import MLPWithBatchNorm
    # NUM_HIDDEN_LAYERS = 8
    # NODES_PER_LAYER = 300
    # model = MLPWithBatchNorm(
    #     input_size=INPUT_SIZE,
    #     output_size=OUTPUT_SIZE,
    #     num_hidden_layers=NUM_HIDDEN_LAYERS,
    #     nodes_per_layer=NODES_PER_LAYER,
    #     dropout_rate=0.05
    # )
    # model.to(DEVICE)

    # Halfing model
    # from model import HalfingModel
    # model = HalfingModel(
    #     input_size=INPUT_SIZE,
    #     output_size=OUTPUT_SIZE,
    #     factor=20,
    #     num_blocks=3,
    #     dropout_rate=0
    # )
    # model.to(DEVICE)

    ############################
    # Loss, optimizer, scheduler
    ############################

    # loss function -> RMSE
    from torch import nn
    # loss_fn = nn.MSELoss(reduction='mean')
    # loss_fn = nn.L1Loss(reduction='mean')

    # SmoothL1Loss (or Huber loss):
    # If the abs difference between the target and the input is small (less than 1), it calculates a half MSE.
    # If the abs difference between the target and the input is large (greater than or equal to 1), it calculates an MAE
    # advantage: -> this loss is more robust to outliers
    loss_fn = nn.SmoothL1Loss(reduction='mean')

    # optimizer -> Adam
    from torch import optim
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # scheduler -> cosine annealing with warm restarts
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(NUM_EPOCHS*len(train_data_loader)*0.03),
        T_mult=2,
        eta_min=LEARNING_RATE * 1e-4,
    )

    # scheduler -> Polynomial learning rate scheduler
    # scheduler visualized: https://www.researchgate.net/publication/224312922/figure/fig1/AS:668980725440524@1536508842675/Plot-of-Q-1-of-our-upper-bound-B1-as-a-function-of-the-decay-rate-g-for-both.png
    from schedulers import PolynomialLRDecay
    MAX_ITER = int(len(train_data_loader) * NUM_EPOCHS - (len(train_data_loader) * WARMUP_EPOCHS))
    print('Polynomial learning rate scheduler - MAX_Iter (number of iterations until decay):', MAX_ITER)
    POLY_POWER = 1.3  # specify the power of the polynomial, 1.0 means linear decay, and 2.0 means quadratic decay
    scheduler = PolynomialLRDecay(optimizer=optimizer,
                                  max_decay_steps=MAX_ITER,  # when to stop decay
                                  end_learning_rate=LEARNING_RATE * 1e-3,
                                  power=POLY_POWER)


    # GradualWarmupScheduler
    from schedulers import GradualWarmupScheduler
    WARMUP_EPOCHS = int(NUM_EPOCHS*len(train_data_loader)*0.03)
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1,
                                       total_epoch=WARMUP_EPOCHS, # when to stop warmup
                                       after_scheduler=scheduler,
                                       is_batch = True)

    # print model summary using torchsummary
    from torchinfo import summary
    tensor_shape = (BATCH_SIZE, INPUT_SIZE)
    summary(model, input_size=tensor_shape, device=DEVICE)  # Update the input_size to (BATCH_SIZE, INPUT_SIZE)

    # print
    print(f"Device: {DEVICE}")

    ############################
    # train model
    ############################
    # import train function
    from train import train_fn, eval_fn
    from utils import save_checkpoint
    from tqdm import trange

    best_test_loss = float('inf')  # initialize with a high value

    # Initialize an empty DataFrame for storing metrics
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "test_loss", "test_rmse", "test_mae"])

    # Train and evaluate the model
    progress_bar = trange(NUM_EPOCHS)
    for epoch in progress_bar:
        # training
        train_loss = train_fn(train_data_loader, model, loss_fn, DEVICE, optimizer, scheduler)

        # testing
        test_loss, mse, rmse, mae = eval_fn(test_data_loader, model, loss_fn, DEVICE)

        # Update the progress bar with the current epoch loss
        progress_bar.set_postfix({"train_loss": f"{format(train_loss, ',.2f')}"})

        # log metrics
        data = pd.DataFrame(
            {"epoch": [epoch], "train_loss": [train_loss], "test_loss": [test_loss], "test_rmse": [rmse],
             "test_mae": [mae]})
        metrics_df = pd.concat([metrics_df, data], ignore_index=True)

        if epoch % 5 == 0:
            # print training loss and test metrics:
            print(
                f"Epoch: {epoch}, Train Loss: {format(train_loss, ',.2f')}, Test Loss:{format(test_loss, ',.2f')}, "
                f"Test RMSE: {format(rmse, ',.2f')}, Test MAE: {format(mae, ',.2f')}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save metrics DataFrame to a CSV file
            cwd = os.getcwd()
            metrics_df.to_csv(os.path.join(cwd, 'models', 'metrics.csv'), index=False)

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            model_path = save_checkpoint(checkpoint, model_name="best_model")

if __name__ == '__main__':
    main()