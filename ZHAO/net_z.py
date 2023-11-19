#Imports
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import numpy as np


# Création du modèle
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*224*224, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = x.float()
        #print("Input data type:", x.dtype)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# model = NeuralNetwork().to(device)
# print(model)