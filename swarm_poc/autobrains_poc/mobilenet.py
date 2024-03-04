# ## SwarmOne
# Please insert your **API key** <br/>
# You can find it in your [Workspace](https://dashboard.swarmone.ai/workspace)

from swarm_one.pytorch import Client

swarm_one_client = Client(api_key="API_KEY")

# ## Imports 📦

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L

import numpy as np
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import datasets


class CustomDataset(Dataset):
    def __init__(self, post_shape, split, num_samples=100):
        self.post_shape = post_shape[1:]
        self.num_samples = num_samples
        self.cifar10 = datasets.CIFAR10(root="../data", train=(split == "train"), download=False)

        if split == "train":
            transform_list = []
            transform_list.extend([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                v2.RandomAdjustSharpness(sharpness_factor=2),
                v2.RandomResizedCrop(self.post_shape, scale=(0.2, 0.8), ratio=(0.5, 1.08)),
                v2.ToDtype(torch.float32, scale=True),
                v2.ToTensor()
            ])
            self.transform_func = Compose(transform_list)
        else:
            self.transform_func = Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.ToTensor()
            ])

    def transforms(self, item):
        image, label = item
        if self.transform_func is not None:
            image = self.transform_func(image)
        return image, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x, y = self.cifar10[idx % len(self.cifar10)]
        x = x.resize(self.post_shape)

        x, y = self.transforms((x, y))

        return x, y


train_dataset = CustomDataset(post_shape=(3, 224, 224),
                              split="train",
                              num_samples=270_000)

val_dataset = CustomDataset(post_shape=(3, 224, 224),
                            split="val",
                            num_samples=30_000)

# ## DataLoaders

train_dataloader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True)

val_dataloader = DataLoader(val_dataset,
                            batch_size=64,
                            shuffle=False)


class MobileNetV2(L.LightningModule):
    def __init__(self, num_classes, optimizer_type, learning_rate):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier = torch.nn.Sequential(torch.nn.Linear(self.model.last_channel, num_classes))

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.view(-1)

        logits = self(images)

        loss = F.cross_entropy(logits, labels.long())
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)

        self.log("train_accuracy", acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch

        val_labels = val_labels.view(-1)

        val_logits = self(val_images)

        val_loss = F.cross_entropy(val_logits, val_labels.long())
        val_preds = torch.argmax(val_logits, dim=1)
        val_acc = self.accuracy(val_preds, val_labels)

        self.log("val_accuracy", val_acc)
        self.log('val_loss', val_loss)
        return val_loss


# ## Models HPO

models = {
    "adam_lr_0.01": MobileNetV2(num_classes=10, optimizer_type="adam", learning_rate=0.01),
    "adam_lr_005": MobileNetV2(num_classes=10, optimizer_type="adam", learning_rate=0.005),
    "adamw_lr_005": MobileNetV2(num_classes=10, optimizer_type="adamW", learning_rate=0.005),
}

hyperparameters = {
    "max_epochs": [10, 20],
    "batch_sizes": [128, 256],
}

# ## Training ⚡⚡⚡
# You can abort any **JOB** / **TASK** at anytime to reduce your costs & total training time - from **your dashboard**

job_id = swarm_one_client.fit(
    model=models,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    hyperparameters=hyperparameters,
    name=f'AutoBrains MobileNetV2',
)

# ## Tensorboard Logs 📉 📈

swarm_one_client.download_tensorboard_logs(job_id, log_dir="path", show_tensorboard=False)

# ## Trained Model Downloading 🎯 🏆
# You can choose to load the model directly to your code, and also download it

trained_model = swarm_one_client.download_trained_model("TASK_ID", ".")

print(trained_model)

history = swarm_one_client.get_task_history("TASK_ID")

print(history)

job_history = swarm_one_client.get_job_history(job_id=job_id, to_pandas=True)

print(job_history)

job_info = swarm_one_client.get_job_information(job_id, to_pandas=True)

print(job_info)
