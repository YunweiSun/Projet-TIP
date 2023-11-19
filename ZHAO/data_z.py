#create csv file for test
import os
import csv
# img_dir="../isic-2020-resized/test-resized"
# with open('test-name.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     field = ["name"]
#     writer.writerow(field)
#     for x in os.listdir(img_dir):
#         writer.writerow([x])


#Imports
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import numpy as np



#Chargement des données
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms

# Define data augmentation transforms
data_augmentation = transforms.Compose([
	transforms.RandomRotation(10),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
])



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+".jpg")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class CustomImageTestset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_name = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name.iloc[idx,0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
    
train_dataset=CustomImageDataset(annotations_file='../isic-2020-resized/train-labels.csv',img_dir='../isic-2020-resized/train-resized')

# Assuming train_dataset contains all your data
total_size = len(train_dataset)

# Create an addtion class 1 from all data
class1_samples = [train_dataset[i] for i in range(total_size) if train_dataset[i][1] == 1]

# Apply data augmentation to class 1 samples and add to train_dataset
train_dataset += 10*[(data_augmentation(sample[0]), 1) for sample in class1_samples]
total_size = len(train_dataset)

# Calculate the size of the validation set (1/5 of the total size)
val_size = total_size // 5

# Calculate the size of the training set
train_size = total_size - val_size

# Use random_split to create datasets for training and validation
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


# test_dataset=CustomImageTestset(annotations_file='../isic-2020-resized/test-name.csv',img_dir='../isic-2020-resized/test-resized')

# for i, sample in enumerate(train_dataset):
#     print(i, sample[0].shape, sample[1])
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample[0].T)
#     if i == 3:
#         plt.show()
#         break
        
# for i, sample in enumerate(val_dataset):
#     print(i, sample[0].shape, sample[1])
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Val Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample[0].T)
#     if i == 3:
#         plt.show()
#         break
        
# for i, sample in enumerate(test_dataset):
#     print(i, sample.shape)
#     ay = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ay.set_title('Test sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample.T)
#     if i == 3:
#         plt.show()
#         break


#Mise en forme des données

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,num_workers=2)

# for X in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     #print(f"Shape of y: {y.shape} {y.dtype}")
#     break
for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(f"batch:{batch_size}")
    break