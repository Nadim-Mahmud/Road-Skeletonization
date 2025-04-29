import torch
from models.skeleton_unet import SkeletonUNet
from engine.tester import Tester
from utils.dataloader import split_dataset_and_loaders
from datasets.road_skeleton_dataset import RoadSkeletonDataset
from transforms.transforms import train_transform
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = RoadSkeletonDataset(config.root_path, transform=train_transform)
_, _, test_loader = split_dataset_and_loaders(dataset, batch_size=config.batch_size)

model = SkeletonUNet().to(device)
model.load_state_dict(torch.load('best_model.pth'))

tester = Tester(model, test_loader, device, threshold=config.threshold_test)
metrics = tester.test()
print(metrics)
tester.visualize_samples()