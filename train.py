import torch
from datasets.road_skeleton_dataset import RoadSkeletonDataset
from transforms.transforms import train_transform
from models.skeleton_unet import SkeletonUNet
from models.loss import WeightedFocalLoss
from engine.trainer import Trainer
from utils.dataloader import split_dataset_and_loaders
import config

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = RoadSkeletonDataset(config.root_path, transform=train_transform)
train_loader, val_loader, test_loader = split_dataset_and_loaders(dataset, batch_size=config.batch_size)

model = SkeletonUNet().to(device)
trainer = Trainer(model, train_loader, val_loader, device, threshold=config.threshold_train)
trainer.fit(epochs=100, early_stop=5)