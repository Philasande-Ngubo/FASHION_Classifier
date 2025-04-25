# Name: Philasande Ngubo
# Date: 20 April 2025
# Custom Nueral Network



import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Data_DIR = "."
    train_data = datasets.FashionMNIST(Data_DIR, train = True, download = False)
    test_data = datasets.FashionMNIST(Data_DIR, train = False, download = False)

    labels = ['T-shirts/ top', 'Trouser', 'Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker', 'Bag', 'Ankle boot']


if __name__ == "__main__":
    main()