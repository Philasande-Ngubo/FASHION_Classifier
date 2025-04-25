# Name: Philasande Ngubo
# Date: 20 April 2025
# Custom Nueral Network

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

INPUT_SIZE = 784
HIDDEN_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
Data_DIR = "."
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = ['T-shirts/ top', 'Trouser', 'Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker', 'Bag', 'Ankle boot']

def main():
    
    train_data = datasets.FashionMNIST(Data_DIR, train = True, download = False, transform = transforms.ToTensor())
    test_data = datasets.FashionMNIST(Data_DIR, train = False, download = False, transform = transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader( dataset=train_data,batch_size=BATCH_SIZE, shuffle= True)
    test_loader = torch.utils.data.DataLoader( dataset=test_data,batch_size=BATCH_SIZE, shuffle= False)


if __name__ == "__main__":
    main()