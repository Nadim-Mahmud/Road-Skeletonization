from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(90, 90)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])