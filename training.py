import torch
import torch.optim as optim
import torch.nn as nn
from dataLoader import train_loader, test_loader
from neutralNetwork import ImageClassifier



device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")

net = ImageClassifier()
net.to(device)

loss_func = nn.CrossEntropyLoss()  # loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer


def test_loop(train_loader, net, loss_func):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        net.eval()  # evaluation mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            outputs = net(images)

            # calculate average loss
            loss = loss_func(outputs, labels) 
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Average loss: %.3f' % (total_loss / len(train_loader)))
    print('Accuracy: %d %%' % (100 * correct / total))

    # net.eval()
    # total = 0
    # size = len(train_loader.dataset)
    # total_loss = 0
    # correct = 0

    # with torch.no_grad():
    #     for X, y in train_loader:
    #         pred = net(X)
    #         test_loss += criterion(pred, y).item()
    #         correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    #         avgLoss = total_loss / len(train_loader)
    #         accuracy = 100 * correct / size

    # print('Average loss: %.3f' % (avgLoss))
    # print('Accuracy: %d %%' % (accuracy))
    # test_loss /= len(train_loader)
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for epoch in range(10):  
    running_loss = 0.0
    for batch, data in enumerate(train_loader, 0):
        # data: list[inputs,labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(labels.size())
        # print(inputs,labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        # print(outputs.size())
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

    
        running_loss += loss.item()
        if batch % 500 == 499: 
            print('[Epoch %d, Batch %4d] loss: %.3f' % (epoch + 1, batch + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')

test_loop(train_loader, net, loss_func)
