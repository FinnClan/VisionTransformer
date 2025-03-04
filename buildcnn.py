import torch.nn as nn

def BuildCnn(input_channels=2, # Number of input channels (e.g., 3 for RGB, 1 for grayscale)
             conv_layers=[(32, 3, 1), (64, 3, 1)], # Configuration for convolutional layers. Each tuple specifies
                                # (out_channels, kernel_size, stride
             use_pooling=True): # Whether to include max-pooling after each convolutional layer.

    layers = []  # List to hold layers of the CNN
    in_channels = input_channels  # Track the number of channels between layers

    # Loop through each specified convolutional layer configuration
    for (out_channels, kernel_size, stride) in conv_layers:
        # Add a convolutional layer with specified parameters
        layers.append(nn.Conv2d(
            in_channels=in_channels,  # Number of input channels from previous layer
            out_channels=out_channels,  # Number of output channels (filters)
            kernel_size=kernel_size,  # Size of the convolutional kernel
            stride=stride,  # Step size of the kernel
            padding=kernel_size // 2  # Maintain spatial dimensions using same-padding
        ))

        # Apply ReLU activation function
        layers.append(nn.ReLU())

        # Optionally, add max pooling to reduce spatial dimensions
        if use_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample by a factor of 2

        # Update input channels for the next layer
        in_channels = out_channels

    # Return the constructed CNN as a sequential model
    return nn.Sequential(*layers)

# Example usage:
# cnn = BuildCnn(input_channels=3, conv_layers=[(32, 3, 1), (64, 3, 1)], use_pooling=True)
