from swarm_one.pytorch import Client

swarm_one_client = Client(api_key="bSNxj8I21w")

import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms


# Define a PyTorch Lightning module
class SimpleNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss


# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Initialize and train the model using PyTorch Lightning
model = SimpleNN()

hyperparameters = {
    "max_epochs": [5, 10, 15],
    "batch_sizes": [64]
}

# Train the model

job_id = swarm_one_client.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    hyperparameters=hyperparameters,
    name="MNIST test"
)

swarm_one_client.download_tensorboard_logs(job_id, log_dir="path", show_tensorboard=False)
