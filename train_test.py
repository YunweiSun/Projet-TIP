
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import csv
# import matplotlib.pyplot as plt

from loadData import train_loader, valid_loader, n_splits, file_name, test_loader
# from loadData_copy import train_loader, valid_loader, n_splits, file_name, test_loader
from network import resnet50, ResNet
from ZHAO.net_z import NeuralNetwork


batch_nbr = 500
epochs = 50
lr = 0.001
momentum = 0.9

# set net model
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")

net = resnet50()
# net = ResNet()
# net = NeuralNetwork()
net.to(device)

# nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1.0)
loss_func = nn.CrossEntropyLoss()  # loss function
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) # optimizer
# optimizer = optim.Adagrad(net.parameters(), lr=lr)
# optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

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

    # 写入内容到文件
    file.write('Average loss: %4.3f    ' % (total_loss / len(train_loader)))
    file.write('Accuracy class[0]: %4.3f %% [%4d/%d]    ' % (100 * correct_0 / total_0, correct_0, total_0))
    file.write('Accuracy class[1]: %4.3f %% [%4d/%d]\n' % (100 * correct_1 / total_1, correct_1, total_1))

    print('Average loss: %.3f    ' % (total_loss / len(train_loader)), end = '')
    print('Accuracy class[0]: %4.3f %% [%4d/%d]    ' % (100 * correct_0 / total_0, correct_0, total_0), end = '')
    print('Accuracy class[1]: %4.3f %% [%4d/%d]    ' % (100 * correct_1 / total_1, correct_1, total_1))

def test(test_loader, net):
    estimated_test_csv = "./isic-2020-resized/test_estimated.csv"
    print("---Test Start---")
    img_label_ls = []
    with torch.no_grad():
        net.eval()  # evaluation mode
        for images, img_name in test_loader:
            # print(f"image_name = {img_name}")
            images = images.to(device)

            outputs = net(images) # outputs = [[a, b]...], a 表示class 0的概率，b表示1的概率

            _, predicted = torch.max(outputs, 1)
            # print(f"estimaed labels = {predicted}")
            img_label = [img_name[0], predicted.item()]
            # print(f"img_label = {img_label}")

            img_label_ls.append(img_label)

    # print(img_label_ls)
    with open(estimated_test_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'target']) 
        for img_label in img_label_ls:
            csv_writer.writerow(img_label)

    print("test fini !!!")



if __name__ == '__main__':

    with open(file_name, 'a') as file:
        file.write(f"loss function: {loss_func}\n")
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
        for i in range(1):
            file.write(f"------dataset: {i}------\n")
            print(f"------dataset: {i}------")
            train_data = train_loader[i]
            test_data = valid_loader[i]
            for epoch in range(epochs): 
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
                        validate(test_data, net, loss_func, file)
                        running_loss = 0.0

                    nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1.0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # validate(test_data, net, loss_func)

        file.write('\nFinished Training\n\n\n')
        print('\nFinished Training\n\n')

        file.write('\nFinished Training\n\n\n')
        print('\nFinished Training\n\n')
        test(test_loader, net)






