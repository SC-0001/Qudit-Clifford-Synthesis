"""
Defines a custom Convolutional Neural Network (CNN) for feature extraction
in the Clifford synthesis reinforcement learning task.
"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn
from math import floor

# Main #######################################################################################

class CustomCliffordCNN(BaseFeaturesExtractor):
    """
    A custom CNN feature extractor that processes observations from the environment.
    The network uses an embedding layer to handle discrete inputs, followed by
    a convolutional layer and 2 fully-connected layers.
    """
    def __init__(
            self, 
            observation_space: gym.spaces.Box, 
            features_dim: int, 
            conv_channels: int, 
            fc1_units: int, 
            num_lvs: int, 
            embedding_dim: int
        ):
        """
        Args:
            observation_space (gym.spaces.Box): The environment's observation space.

            features_dim (int): The number of features to extract (output dimension).

            conv_channels (int): The number of output channels for the convolutional layer.

            fc1_units (int): The number of units in the first fully connected layer.

            num_lvs (int): The number of energy levels per qudit, which defines the
                vocabulary size for the embedding layer.

            embedding_dim (int): The dimensionality of the embedding vectors.
        """
        super().__init__(observation_space, features_dim)

        # Set device
        if torch.cuda.is_available():
            torch_device = "cuda"
        # elif torch.backends.mps.is_available():
            # torch_device = "mps"
        else:
            torch_device = "cpu"
        self.device = torch.device(torch_device)

        # Observation shape: (4, num_qudits, num_qudits)
        self.num_qudits = observation_space.shape[1]

        # Embedding layer
        # Used instead of one-hot encoding to reduce input sparsity.
        # Input shape: (batch_size, 4, num_qudits, num_qudits)
        # Output shape: (batch_size,  4, num_qudits, num_qudits, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings  = num_lvs , embedding_dim = embedding_dim)
        self.embedding_dim = embedding_dim

        # Convolutional layer
        # The hope is that the model learns something about coupling maps. 
        # Input shape: (batch_size, 4 * embedding_dim, num_qudits, num_qudits)
        # Output shape: (batch_size, conv_channels, l, l)
        padding = 0
        stride = 1
        kernel_size = 2
        l = floor((self.num_qudits + 2*padding - kernel_size) / stride) + 1
        self.conv = nn.Sequential(
            nn.Conv2d(4 * embedding_dim, conv_channels, kernel_size = kernel_size, padding = padding, stride = stride),
            nn.ReLU()
        ).to(self.device)
        
        # Fully connected layers
        # Input shape: (batch_size, conv_channels * l * l)
        # Output shape: (batch_size, features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(conv_channels * l * l, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, features_dim),
            nn.ReLU()
        ).to(self.device)

        self.init_weights()

    def init_weights(self):
        """
        Initializes weights using Xavier uniform initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                assert m.bias is not None
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the feature extractor.

        Args:
            observations (torch.Tensor): The input observation tensor from the environment.

        Returns:
            final_features (torch.Tensor): The extracted features tensor.
        """
        # Input shape: (batch_size, 4, num_qudits, num_qudits)
        observations = observations.to(self.device).int()
        batch_size = observations.shape[0]
        
        # Current Shape: (batch_size, 4, num_qudits, num_qudits, embedding_dim)
        embedded_observations = self.embedding(observations)

        # Current Shape: (batch_size, embedding_dim, 4, num_qudits, num_qudits)
        permuted_observations = embedded_observations.permute(0, 4, 1, 2, 3).contiguous()

        # Current Shape: (batch_size, embedding_dim*4, num_qudits, num_qudits)
        reshaped_observations = permuted_observations.view(batch_size, self.embedding_dim * 4, self.num_qudits, self.num_qudits)

        convoluted_observations = self.conv(reshaped_observations)
        flattened_observations = convoluted_observations.flatten(1)
        final_features = self.mlp(flattened_observations)
        return final_features

# End of File ###############################################################################