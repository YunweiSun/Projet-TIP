import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # csv_file : path to csv file
        # root_dir : path to image file
    
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        # label = self.annotations.iloc[idx, 1]
        label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)

        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [img for img in os.listdir(root_dir) if img.endswith(('.jpg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# create dataset
train_dataset = TrainDataset(csv_file='isic-2020-resized/train-labels.csv', root_dir='isic-2020-resized/train-resized', transform=transform)
test_dataset = TestDataset(root_dir='isic-2020-resized/test-resized', transform=transform)

# create data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




