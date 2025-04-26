# Name: Philasande Ngubo
# Date: 20 April 2025
# Custom Nueral Network

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

INPUT_SIZE = 784
HIDDEN_SIZE = 400
NUM_CLASSES = 10
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 100
Data_DIR = "."
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = ['T-shirts/ top', 'Trouser', 'Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker', 'Bag', 'Ankle boot']


class ForwardNueralNetwork(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(ForwardNueralNetwork,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size )
        self.relu = nn.ReLU()
        self.l2 =nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        return self.l2(  self.relu( self.l1(x) ) )

def train_model(model, train_loader, test_loader, criterion, optimizer):
    n_steps = len(train_loader)                     
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch +1} of {NUM_EPOCHS}')
        for i, (x, true_labels) in enumerate(train_loader):
            x = x.reshape(-1, 28*28).to(DEVICE)
            true_labels = true_labels.to(DEVICE)

            predicted_labels = model(x)
            loss = criterion(predicted_labels, true_labels)

            optimizer.zero_grad()
            loss.backward()   #back propagation

            optimizer.step()

            if ((i+1) % 200 == 0):
                print(f'  Loss = {loss.item():.4f}')

def main():
    
    print("Loading dataset...")
    train_data = datasets.FashionMNIST(Data_DIR, train = True, download = False, transform = transforms.ToTensor())
    test_data = datasets.FashionMNIST(Data_DIR, train = False, download = False, transform = transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader( dataset=train_data,batch_size=BATCH_SIZE, shuffle= True)
    test_loader = torch.utils.data.DataLoader( dataset=test_data,batch_size=BATCH_SIZE, shuffle= False)

    model = ForwardNueralNetwork(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_model(model = model , train_loader = train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer)

    # training 
    
    
    with torch.no_grad():
        num_correct_predictions = 0
        testing_data_size = 0
        for i, (x, true_labels) in enumerate(test_loader):
             x = x.reshape(-1, 28*28).to(DEVICE)
             true_labels = true_labels.to(DEVICE)

             predicted_labels = model(x)

             _, predictions = torch.max(predicted_labels, 1)
             testing_data_size += true_labels.shape[0]
             num_correct_predictions += (predictions == true_labels).sum().item()
        
        model_accuracy = 100.0 * ( num_correct_predictions/ testing_data_size)
        print(f'Model accurracy {model_accuracy:.2f}%')





if __name__ == "__main__":
    main()

