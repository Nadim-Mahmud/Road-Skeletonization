import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RoadSkeletonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.highway_types = {
            'motorway': 0, 'trunk': 1, 'primary': 2,
            'secondary': 3, 'tertiary': 4, 'residential': 5,
            'service': 6, 'footway': 7, 'cycleway': 8
        }
        self.samples = []
        thinning_dir = self.root_dir / 'thinning'
        for img_path in thinning_dir.glob('*.png'):
            stem = img_path.stem
            target_path = self.root_dir / 'targets_png' / f'{stem}.png'
            geojson_path = self.root_dir / 'targets_geojson' / f'{stem}.geojson'
            if target_path.exists() and geojson_path.exists():
                self.samples.append((img_path, target_path, geojson_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target_path, geojson_path = self.samples[idx]
        image = Image.open(img_path).convert('L')
        target = Image.open(target_path).convert('L')
        target = target.point(lambda x: 0 if x < 128 else 255)
        with open(geojson_path) as f:
            geojson = json.load(f)
        properties = geojson['features'][0]['properties']
        lanes = int(properties.get('lanes', 1))
        highway_type = self.highway_types.get(
            properties.get('highway', 'residential'), 5
        )
        if self.transform:
            image = self.transform(image)
        target = transforms.ToTensor()(target)
        assert torch.all(torch.logical_or(target == 0.0, target == 1.0)), "Target contains invalid values"
        return image, target, {'lanes': torch.tensor(lanes), 'highway_type': torch.tensor(highway_type)}

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    annotations = [item[2] for item in batch]
    return images, targets, annotations