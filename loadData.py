import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold

dataindex = range(10000)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = []
test_loader = []
n_splits = 8

#建立自己的dataset
class CreateDatasetFromImages(Dataset):
    def __init__(self, csv_info, file_path, data_index = dataindex, resize_height=224, resize_width=224, transform = transform):

        self.resize_height = resize_height
        self.resize_width = resize_width
 
        # csv_path = "./isic-2020-resized/train-labels.csv"
        # train_path = "./isic-2020-resized/train-resized"
        self.file_path = file_path
        self.transform = transform
 
        self.data_info = csv_info 
        # self.data_info = pd.read_csv(csv_path)
        self.data_info = self.data_info.iloc[data_index]

 
    def __getitem__(self, index):
        label = torch.tensor(int(self.data_info.iloc[index, 1]))
        img_as_img = Image.open(self.file_path + self.data_info.iloc[index, 0] + ".jpg")
        img_as_img = self.transform(img_as_img)
        return img_as_img, label  #返回每一个index对应的图片数据和对应的label
        
    def __len__(self):
        return len(self.data_info)-1
    



csv_path = "./isic-2020-resized/train-labels.csv"
data_path = "./isic-2020-resized/train-resized/"

csv_info = pd.read_csv(csv_path)
size_data = len(csv_info)
# print(size_data)

# add more class 1 to the data
nbr_copy = 10

selected_rows = csv_info[csv_info.iloc[:, 1] == 1]

# 复制并添加这些行到DataFrame
for _ in range(nbr_copy): 
    csv_info = pd.concat([csv_info, selected_rows.copy()], ignore_index=True)

size_data += nbr_copy*len(selected_rows)
# print(size_data)



# to seperate train_dataset and test_dataset
data_index = [None]*size_data

kf = KFold(n_splits=n_splits, shuffle=True)

for i, (train_index, test_index) in enumerate(kf.split(data_index)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    
    TrainDataset = CreateDatasetFromImages(csv_info, data_path, train_index)
    TestDataset = CreateDatasetFromImages(csv_info, data_path, test_index)
    train_loader.append(
        torch.utils.data.DataLoader(
            dataset=TrainDataset,
            batch_size=32, 
            shuffle=True,
        ))
    test_loader.append(
        torch.utils.data.DataLoader(
            dataset=TestDataset,
            batch_size=32, 
            shuffle=False,
        ))

    # print(f"length_train = {len(TrainDataset)}")
    # print(f"length_test = {len(TestDataset)}")


