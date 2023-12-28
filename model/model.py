import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .effnet import *
# An example of model, very simple
from torchvision.models import efficientnet_b0  # You can choose other versions like b1, b2, etc.


def MyEffNet(spectral_channels, out_pts):
    blocks_args, global_params = param_template(spectral_channels, out_pts)
    model = BaseEfficientNet(blocks_args, global_params)
    return model


class MyRegressionCNN(nn.Module):
    def __init__(self, spectral_channels, out_pts):
        self.spectral_channels = spectral_channels
        super(MyRegressionCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        # Residual connection
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64)
        )
        # Fully connected layers
        self.fc1 = nn.Linear(self.spectral_channels * 64, 512)  # Adjust the size according to your input length
        self.fc2 = nn.Linear(512, out_pts)

    def forward(self, x):
        # Add channel dimension (batch_size, channels, length)
        x = x.unsqueeze(1)
        # Convolutional layers with residual connection

        x = F.relu(self.bn1(self.conv1(x)))
        identity = self.shortcut(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) + identity  # Adding residual (shortcut connection)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class MyRegressionFCN(nn.Module):
    def __init__(self, spectral_channels, out_pts):
        super(MyRegressionFCN, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(spectral_channels, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, out_pts)

        # Define residual connections
        self.shortcut1 = nn.Linear(spectral_channels, 128)
        self.shortcut2 = nn.Linear(128, 32)

    def forward(self, x):
        # Initial input layer
        identity = x

        # First hidden layer with ReLU
        x = F.relu(self.fc1(x))

        # Second hidden layer with residual connection and ReLU
        identity1 = self.shortcut1(identity)
        x = F.relu(self.fc2(x) + identity1)

        # Third hidden layer with ReLU
        x = F.relu(self.fc3(x))

        # Fourth hidden layer with residual connection and ReLU
        identity2 = self.shortcut2(identity1)
        x = F.relu(self.fc4(x) + identity2)

        # Output layer with sigmoid activation
        # x = torch.sigmoid(self.fc5(x))
        x = self.fc5(x)
        return x


if __name__ == '__main__':
    #blocks_args, global_params = gen_param(20)
    model = MyRegressionCNN(64, 1)
    from torchsummary import summary
    summary(model, (64,), 32, 'cpu')