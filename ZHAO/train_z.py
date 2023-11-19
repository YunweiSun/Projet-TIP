#Imports
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import numpy as np

from data_z import train_dataloader, val_dataloader, train_dataset
from net_z import model, device

#défini la fonction de perte et l'optimiseur
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

#training
epochs=5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
#         print(pred)
        
#         print("Target labels:", y)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        optimizer.zero_grad()

        if batch % 6400 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
            # Validation
            
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            total_samples_0 = 0
            total_samples_1 = 0
            correct_predictions_0 = 0
            correct_predictions_1 = 0

            with torch.no_grad():
                model.eval()  # Set the model to evaluation mode
                for val_batch, (val_X, val_y) in enumerate(val_dataloader):
                    val_X, val_y = val_X.to(device), val_y.to(device)

                    val_pred = model(val_X)
                    val_loss += loss_fn(val_pred, val_y).item()

                    _, predicted_labels = torch.max(val_pred, 1)
                    correct_predictions_0 += ((val_y == 0) & (predicted_labels == 0)).sum().item()
                    correct_predictions_1 += ((val_y == 1) & (predicted_labels == 1)).sum().item()
                    
                    # Calculate accuracy for each class
                    total_samples_0 += (val_y == 0).sum().item ()
                    total_samples_1 += (val_y == 1).sum().item ()

            accuracy_0 = correct_predictions_0 / total_samples_0
            accuracy_1 = correct_predictions_1 / total_samples_1

            avg_val_loss = val_loss / (val_batch + 1)
            print(f"Validation loss: {avg_val_loss:.4f}")
            # Display class-wise accuracy
            print(f"        Accuracy for class 0: {accuracy_0:.4f}    class 1: {accuracy_1:.4f}")
            print()
            
    
print("Done!")