# ## SwarmOne
# Please insert your **API key** <br/>
# You can find it in your [Workspace](https://dashboard.swarmone.ai/workspace)

from swarm_one.pytorch import Client

swarm_one_client = Client(api_key="API_KEY")

# ## Imports 📦
import lightning as L
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset

DATASET_PATH = os.environ.get("PATH_DATASETS", "../stl10_data/")


# ## Contrastive Transformations
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


# ## Datasets
# Function to create a subset
def create_subset(dataset, subset_size=4096):
    total_samples = len(dataset)
    indices = np.random.choice(range(total_samples), subset_size, replace=True)
    return Subset(dataset, indices)


unlabeled_data_full = STL10(
    root=DATASET_PATH,
    split="unlabeled",
    download=False,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
)

train_data_contrast_full = STL10(
    root=DATASET_PATH,
    split="train",
    download=False,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
)

unlabeled_data = create_subset(unlabeled_data_full, subset_size=9000)
train_data_contrast = create_subset(train_data_contrast_full, subset_size=1000)

# ## DataLoaders
train_dataloader = DataLoader(
    unlabeled_data,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)
val_dataloader = DataLoader(
    train_data_contrast,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)


# ## DinoV2 LightningModule
class DinoV2Model(L.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        # self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=True)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        self.lr = learning_rate
        self.temperature = 0.07

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def info_nce_loss(self, batch, mode="train"):
        lst, _ = batch
        x1, x2 = lst
        imgs = torch.cat([x1, x2], dim=0)

        # Encode all images
        feats = self.model(imgs)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


model = DinoV2Model(learning_rate=0.01)

# ## Training ⚡⚡⚡
# You can abort any **JOB** / **TASK** at anytime to reduce your costs & total training time - from **your dashboard**
job_id = swarm_one_client.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    hyperparameters={
        "max_epochs": [10],
        "batch_sizes": [2048],
    },
    name=f'AutoBrains Dino V2 Giant'
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
