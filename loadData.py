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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    transforms.RandomRotation(10),  # 随机旋转
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度，对比度
])

train_loader = []
valid_loader = []
n_splits = 8
n_splits_1 = 8
batch_size = 10
size_data = 0

csv_path = "./isic-2020-resized/train-labels.csv"
data_path = "./isic-2020-resized/train-resized/"
test_path = "./isic-2020-resized/test-resized/"
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
    
            # csv_path = "./isic-2020-resized/train-labels.csv"
            # train_path = "./isic-2020-resized/train-resized"
            self.file_path = file_path
            self.transform = transform
    
            self.data_info = csv_info 
            # self.data_info = pd.read_csv(csv_path)
            # self.data_info = self.data_info.iloc[data_index]

    
        def __getitem__(self, index):
            label = torch.tensor(int(self.data_info.iloc[index, 1]))
            img_as_img = Image.open(self.file_path + self.data_info.iloc[index, 0] + ".jpg")
            img_as_img = self.transform(img_as_img)
            return (img_as_img, label)  #返回每一个index对应的图片数据和对应的label
            
        def __len__(self):
            return len(self.data_info)

    # 处理获取的csv文件，提取出label 1 和label 0

    # # add more class 1 to the data
    # nbr_copy = 10
    # selected_rows = csv_info[csv_info.iloc[:, 1] == 1]
    # # 复制并添加这些行到DataFrame
    # for _ in range(nbr_copy): 
    #     csv_info = pd.concat([csv_info, selected_rows.copy()], ignore_index=True)

    # size_data += nbr_copy*len(selected_rows)

    # find label 1 and diviser en 1:7
    img_1 = csv_info[csv_info.iloc[:, 1] == 1]
    nbr_1 = len(img_1)
    split1 = nbr_1//n_splits_1
    img_1_valid = img_1.sample(n=split1, random_state=2) 
    img_1_train = img_1.drop(img_1_valid.index)
    file.write(f"len_img_1_train = {len(img_1_train)}, len_img_1_valid = {len(img_1_valid)}\n")
    print(f"len_img_1_train = {len(img_1_train)}, len_img_1_valid = {len(img_1_valid)}")

    # extrait tous les 1 dans les img
    img_0 = csv_info[csv_info.iloc[:, 1] == 0]
    size_data = len(img_0)
    file.write(f"data size after extrait = {size_data}\n")
    print(f"data size after extrait = {size_data}")

    # 将csv文件分成1:7
    split0 = size_data//n_splits
    img_valid = img_0.sample(n=split0, random_state=1) 
    img_train = img_0.drop(img_valid.index)
    file.write(f"size train no 1 = {len(img_train)}, size valid no 1 = {len(img_valid)}\n")
    print(f"size train no 1 = {len(img_train)}, size valid no 1 = {len(img_valid)}")

    # 重置索引
    img_train = img_train.reset_index(drop=True)
    img_valid = img_valid.reset_index(drop=True)

    # copy 10 times of img_1_train to img_train
    n_copytrain = 1
    copy_img = [img_1_train.copy() for _ in range(n_copytrain)]
    add_img = pd.concat(copy_img, ignore_index=True)
    img_train = pd.concat([img_train, add_img], ignore_index=True)

    # copy 1 times of img_1_valid to img_valid
    n_copyvalid = 1
    copy_img = [img_1_valid.copy() for _ in range(n_copyvalid)]
    add_img = pd.concat(copy_img, ignore_index=True)
    img_valid = pd.concat([img_valid, add_img], ignore_index=True)
    file.write(f"size train with 1 = {len(img_train)}, size valid with 1 = {len(img_valid)}\n")
    print(f"size train with 1 = {len(img_train)}, size valid with 1 = {len(img_valid)}")

    # TrainDataset = CreateDatasetFromImages(img_train, data_path, transform=transform_train)
    TrainDataset = CreateDatasetFromImages(img_train, data_path)
    # ValidDataset = CreateDatasetFromImages(img_valid, data_path, transform=transform_train)
    ValidDataset = CreateDatasetFromImages(img_valid, data_path)


    file.write(f"size TrainDataset = {len(TrainDataset)}, size ValidDataset = {len(ValidDataset)}\n")
    print(f"size TrainDataset = {len(TrainDataset)}, size ValidDataset = {len(ValidDataset)}")


    train_loader.append(
        torch.utils.data.DataLoader(
            dataset=TrainDataset,
            batch_size=batch_size, 
            shuffle=True,
        ))

    valid_loader.append(
        torch.utils.data.DataLoader(
            dataset=ValidDataset,
            batch_size=batch_size, 
            shuffle=False,
        ))


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
    
    test_csv = pd.read_csv(test_csv_path)
    TestDataset = CreateTestDataset(test_csv, test_path)
        
    test_loader = torch.utils.data.DataLoader(
            dataset=TestDataset,
            shuffle=False,
        )


