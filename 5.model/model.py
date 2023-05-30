"""
Script: model.py
============================
This script contains three different variations of multi-layer perceptron (MLP) models for use in machine learning tasks:

1. MLP: A standard fully-connected neural network model with an optional dropout layer after every two blocks except for the final block.
    This dropout is designed to reduce overfitting.

2. MLPWithBatchNorm: Similar to MLP, but includes batch normalization in each layer. This feature aims to improve the model's speed,
    performance, and stability.

3. HalfingModel: A unique model in which the number of features is halved after each block, thereby potentially reducing the complexity and
    computational cost of the model.

Each model is defined as a class that inherits from PyTorch's nn.Module. Each class contains an initialization function to set up the
model architecture, and a forward function to perform the forward propagation through the model.
"""

from torch import nn

class MLP(nn.Module):
    """
    linear model, fully connected NN, allows for non-linearities via ReLU
    -> built by stacking layer blocks, each block consists of a linear layer followed by a non-linear activation function"""
    def __init__(self, input_size, output_size, num_hidden_layers, nodes_per_layer, dropout_rate=0.1):
        super(MLP, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, nodes_per_layer))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            layers.append(nn.ReLU())

            # Add a dropout layer after every 2 blocks except for the final block
            if (i + 1) % 2 == 0 and i != num_hidden_layers - 2:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(nodes_per_layer, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, nodes_per_layer, dropout_rate=0.1):
        super(MLPWithBatchNorm, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, nodes_per_layer))
        layers.append(nn.BatchNorm1d(nodes_per_layer))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            layers.append(nn.BatchNorm1d(nodes_per_layer))
            layers.append(nn.ReLU())

            # Add a dropout layer after every 2 blocks except for the final block
            if (i + 1) % 2 == 0 and i != num_hidden_layers - 2:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(nodes_per_layer, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class HalfingModel(nn.Module):
    def __init__(self, input_size, output_size, factor = 8, num_blocks=2, dropout_rate=0.1):
        super(HalfingModel, self).__init__()

        layers = []

        in_features = input_size * factor

        # Initial layer
        layers.append(nn.Linear(input_size, in_features))
        layers.append(nn.ReLU())



        for _ in range(num_blocks):
            # Half the number of features after every block
            block = nn.Sequential(
                nn.Linear(in_features, in_features//2),
                nn.BatchNorm1d(in_features//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),


                nn.Linear( in_features//2,  in_features//2),
                nn.BatchNorm1d(in_features//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            layers.append(block)

            in_features = in_features // 2
            print(f"block: {block}, in_features: {in_features}")

        self.blocks = nn.Sequential(*layers)

        # Final block
        # @GPT: do it in the same style as above
        final_block = nn.Sequential(
            nn.Linear(in_features, input_size*2),
            nn.ReLU(),

            nn.Linear(input_size*2, input_size),
            nn.ReLU(),

            nn.Linear(input_size, output_size)
        )
        self.final_block = final_block

    def forward(self, x):
        x = self.blocks(x)
        x = self.final_block(x)
        return x