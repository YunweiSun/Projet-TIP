import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),    
])

n_splits = 8
n_splits_1 = 15
batch_size = 32
size_data = 0

csv_path = "./isic-2020-resized/train-labels.csv"
# data_path = "./isic-2020-resized/train-resized/"
# test_path = "./isic-2020-resized/test-resized/"
data_path = "./isic-2020-resized/train-resized-nohair/"
test_path = "./isic-2020-resized/test-resized-nohair/"
test_csv_path = "./isic-2020-resized/test.csv"

file_name = 'test_data.txt'
try:
    os.remove(file_name)
    print(f'{file_name} delete success。')
except FileNotFoundError:
    print(f'{file_name} do not exist')
except Exception as e:
    print(f'delete {file_name} error: {e}')

with open(file_name, 'a') as file:
    csv_info = pd.read_csv(csv_path)
    size_data = len(csv_info)
    file.write(f"data size init = {size_data}\n")
    print(f"data size init = {size_data}")

    #建立自己的dataset
    # data_index 原本用于区分trans data和valid data
    class CreateDatasetFromImages(Dataset):
        def __init__(self, csv_info, file_path, data_index = range(size_data), resize_height=224, resize_width=224, transform = transform):

            self.resize_height = resize_height
            self.resize_width = resize_width
    
            self.file_path = file_path
            self.transform = transform
    
            self.data_info = csv_info 

    
        def __getitem__(self, index):
            label = torch.tensor(int(self.data_info.iloc[index, 1]))
            img_as_img = Image.open(self.file_path + self.data_info.iloc[index, 0] + ".jpg")
            img_as_img = self.transform(img_as_img)
            return (img_as_img, label)  #返回每一个index对应的图片数据和对应的label
            
        def __len__(self):
            return len(self.data_info)

            
    class CreateTestDataset(Dataset):
        def __init__(self, csv_info, file_path, data_index = range(10982), resize_height=224, resize_width=224, transform = transform):

            self.resize_height = resize_height
            self.resize_width = resize_width

            self.file_path = file_path
            self.transform = transform
            
            self.data_info = csv_info 
            # self.data_info = pd.read_csv(csv_path)
            self.data_info = self.data_info.iloc[data_index]


        def __getitem__(self, index):
            img_name = self.data_info.iloc[index, 0]
            img_as_img = Image.open(self.file_path + img_name + ".jpg")
            img_as_img = self.transform(img_as_img)
            return img_as_img, img_name
            
        def __len__(self):
            return len(self.data_info)

    # 处理获取的csv文件，提取出label 1 和label 0

    # add more class 1 to the data
    nbr_copy = 15
    selected_rows = csv_info[csv_info.iloc[:, 1] == 1]
    for _ in range(nbr_copy): 
        csv_info = pd.concat([csv_info, selected_rows.copy()], ignore_index=True)
    csv_info = csv_info.reset_index(drop=True)

    size_data += nbr_copy*len(selected_rows)

    Dataset = CreateDatasetFromImages(csv_info, data_path, transform=transform)

    torch.manual_seed(42)
    train_data,valid_data=torch.utils.data.random_split(Dataset, [0.9, 0.1])


    train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=batch_size, 
            shuffle=True,
            drop_last = True
        )

    valid_loader = torch.utils.data.DataLoader(
            dataset=valid_data,
            batch_size=batch_size, 
            shuffle=True,
            drop_last = True
        )

    
    test_csv = pd.read_csv(test_csv_path)
    TestDataset = CreateTestDataset(test_csv, test_path)
        
    test_loader = torch.utils.data.DataLoader(
            dataset=TestDataset,
            shuffle=False,
        )


