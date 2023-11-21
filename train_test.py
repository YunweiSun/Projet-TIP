
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import csv
from torchvision import models, transforms
import matplotlib.pyplot as plt
from datetime import datetime

from loadData import train_loader, valid_loader, n_splits, file_name, test_loader, Dataset
from network import resnet50
from resnet18 import resnet18
from resnet18_g import resnet18_g


batch_nbr = 200
epochs = 30
lr = 0.001
# lr = 0.0001
momentum = 0.9
batch_size = 32

# set net model
device = torch.device("cuda:0")

# net = resnet50()
# net = models.resnet18()
# net = resnet18()
net = resnet18_g()
# net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# net = models.resnet50(pretrained = True)
for param in net.parameters():
    param.requires_grad = True  
# net.fc=torch.nn.Linear(2048,2)
# net.fc=torch.nn.Linear(512,2)
net.to(device)


# nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1.0)
weight_0 = 1
weight_1 = 2
weights = torch.FloatTensor([weight_0, weight_1]).to(device)
print(f"weight: 0 = {weight_0}, 1 = {weight_1}")
loss_func = nn.CrossEntropyLoss(weight = weights)  # loss function
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) # optimizer
# optimizer = optim.Adagrad(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def validate(valid_loader, net, loss_func, file):
    file.write(" --> valid: ")
    print(" --> valid: ", end = "")
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    total_loss = 0.0
    with torch.no_grad():
        net.eval()  # evaluation mode
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device) 

            # print(f"\nlabels = {labels}")
            outputs = net(images) # outputs = [[a, b]...], a 表示class 0的概率，b表示1的概率

            # calculate average loss
            loss = loss_func(outputs, labels) 
            total_loss += loss.item()


            _, predicted = torch.max(outputs, 1)
            # total += labels.size(0)
            total_0 += (labels == 0).sum().item()
            total_1 += (labels == 1).sum().item()
            correct_0 += ((labels == 0) & (predicted == 0)).sum().item()
            correct_1 += ((labels == 1) & (predicted == 1)).sum().item()

    loss_moyen = total_loss / len(valid_loader)
    accuracy_0 = correct_0 / total_0
    accuracy_1 = correct_1 / total_1
    # 写入内容到文件
    file.write('Average loss: %4.3f    ' % (loss_moyen))
    file.write('Accuracy class[0]: %4.3f %% [%4d/%d]    ' % (100 * correct_0 / total_0, correct_0, total_0))
    file.write('Accuracy class[1]: %4.3f %% [%4d/%d]\n' % (100 * correct_1 / total_1, correct_1, total_1))

    print('Average loss: %.3f    ' % (total_loss / len(valid_loader)), end = '')
    print('Accuracy class[0]: %4.3f %% [%4d/%d]    ' % (100 * accuracy_0, correct_0, total_0), end = '')
    print('Accuracy class[1]: %4.3f %% [%4d/%d]    ' % (100 * accuracy_1, correct_1, total_1))

    return loss_moyen, accuracy_0, accuracy_1

def test(test_loader, net, epoch):
    estimated_test_csv = f"./isic-2020-resized/test_estimated_{epoch}.csv"
    print("---Test Start---")
    img_label_ls = []
    with torch.no_grad():
        net.eval()  # evaluation mode
        for images, img_name in test_loader:
            # print(f"image_name = {img_name}")
            images = images.to(device)

            outputs = net(images) # outputs = [[a, b]...], a 表示class 0的概率，b表示1的概率

            proba_1 = outputs[0][1]
            img_label = [img_name[0], proba_1.item()]
            img_label_ls.append(img_label)

    # print(img_label_ls)
    with open(estimated_test_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'target']) 
        for img_label in img_label_ls:
            csv_writer.writerow(img_label)

    print("test fini !!!")



if __name__ == '__main__':
    loss_moyen = []
    accuracy_0 = []
    accuracy_1 = []
    with open(file_name, 'a') as file:
        file.write(f"loss function: {loss_func}\n")
        file.write(f"weight: 0 = {weight_0}, 1 = {weight_1}")
        file.write(f"optimizer: {optimizer}\n")
        file.write(f"learning rate = {lr}, momentum = {momentum}\n")
        file.write(f"number of epochs = {epochs}\n")
        file.write(f"return loss after each {batch_nbr} batchs\n")
        file.write("\nStart Training...\n")

        print(f"loss function: {loss_func}")
        print(f"optimizer: {optimizer}")
        print(f"learning rate = {lr}, momentum = {momentum}")
        print(f"number of epochs = {epochs}")
        print(f"return loss after each {batch_nbr} batchs")
        
        print("\nStart Training...")
        
        train_data = train_loader
        valid_data = valid_loader
        for epoch in range(epochs): 
            t_start = datetime.now()
            file.write('-- Epoch %d --\n'%(epoch+1))
            print('-- Epoch %d --'%(epoch+1))
            running_loss = 0.0
            for batch, data in enumerate(train_data, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = loss_func(outputs, labels)

                running_loss += loss.item()
                if batch % batch_nbr == batch_nbr-1: 
                    file.write('Batch %4d, loss: %.3f    ' % (batch + 1, running_loss / batch_nbr))
                    print('Batch %4d, loss: %.3f' % (batch + 1, running_loss / batch_nbr), end = "\t")
                    # validate(valid_data, net, loss_func, file)
                    running_loss = 0.0

                nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t_train_fini = datetime.now()
            print(f"\ntrain time: {t_train_fini-t_start}")

            loss, accu_0, accu_1 = validate(valid_data, net, loss_func, file)
            t_valid_fini = datetime.now()
            print(f"valid time: {t_valid_fini-t_train_fini}")

            loss_moyen.append(loss)
            accuracy_0.append(accu_0)
            accuracy_1.append(accu_1)

            

            if (epoch+1)%5 == 0:
                test(test_loader, net, epoch+1)

                plt.figure(1)
                plt.plot(range(epoch+1), loss_moyen, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss Over Epochs')
                plt.savefig('./plot/Loss.png')

                plt.figure(2)
                plt.plot(range(epoch+1), accuracy_0, marker='o', label = "accuracy_0")
                plt.plot(range(epoch+1), accuracy_1, marker='o', label = "accuracy_1")
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of 0 and 1 Over Epochs')
                plt.legend()
                plt.savefig('./plot/Accuracy.png')

        file.write('\nFinished Training\n\n\n')
        print('\nFinished Training\n\n')


        






