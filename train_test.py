
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from loadData import train_loader, test_loader, n_splits
from network import resnet50


# set net model
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")

net = resnet50()
net.to(device)

# nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1.0)
loss_func = nn.CrossEntropyLoss()  # loss function
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9) # optimizer

batch_size = 100

def validate(test_loader, net, loss_func):
    print("--test: ", end = "")
    correct_0 = 0
    correct_1 = 0
    # total = 0
    total_0 = 0
    total_1 = 0
    total_loss = 0.0
    with torch.no_grad():
        net.eval()  # evaluation mode
        for images, labels in test_loader:
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
    

    print('Average loss: %.3f' % (total_loss / len(train_loader)), end = '\t')
    print('Accuracy class[0]: %.3f %% [%5f/%f]' % (100 * correct_0 / total_0, correct_0, total_0), end = '\t')
    print('Accuracy class[1]: %.3f %% [%5f/%f]' % (100 * correct_1 / total_1, correct_1, total_1))

if __name__ == '__main__':
    print("Start Training...\n")
    for i in range(1):
        print(f"------dataset: {i}------")
        train_data = train_loader[i]
        test_data = test_loader[i]
        for epoch in range(3): 
            print('-- Epoch %d --'%(epoch+1))
            running_loss = 0.0
            for batch, data in enumerate(train_data, 0):
                # data: list[inputs,labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # print(labels.size())
                # print(inputs,labels)

                outputs = net(inputs)
                # print(outputs.size())
                loss = loss_func(outputs, labels)

                running_loss += loss.item()
                if batch % batch_size == batch_size-1: 
                    print('Batch %4d, loss: %.3f' % (batch + 1, running_loss / batch_size), end = "\t")
                    validate(test_data, net, loss_func)
                    running_loss = 0.0

                nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # validate(test_data, net, loss_func)

    print('\nFinished Training\n\n')





