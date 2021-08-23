#a simple model for MNIST 

import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #functions which dont have arguments like activation functions
from torch.utils.data import DataLoader #easier dataset management

import torchvision.datasets as datasets 
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__() #calls initialisation method of parent class
        self.fc1 = nn.Linear(input_size, 60)
        self.fc2= nn.Linear(60,80)
        self.fc3 = nn.Linear(80,num_classes)

    def forward(self,x):
        x= F.relu(self.fc1(x))
        x= self.fc2(x)
        x= self.fc3(x)
        return x
model = NN(784,10) #10 digits in mnist 784 =28*28
x= torch.randn(64,784)
print(model(x).shape)


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_size = 784
num_classes=10
learning_rate =0.001
batch_size =64
num_epochs =1

#load data
train_dataset = datasets.MNIST(root ='dataset/',train=True,transform = transforms.ToTensor(),download=False)
train_loader = DataLoader(dataset = train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root ='dataset/',train=False,transform = transforms.ToTensor(),download=False)
test_loader = DataLoader(dataset = test_dataset,batch_size=batch_size,shuffle=True)


#initialise model 
model = NN(
    input_size=input_size,
    num_classes=num_classes
).to(device)

#loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


#Train network
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):

        #get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        #get data to correct shape
        data=data.reshape(data.shape[0],-1)
        #print(data.shape)

        #forward
        scores= model(data)
        loss= criterion(scores,targets)

        #backward
        optimizer.zero_grad()#to set the gradients calculated in previous batches to zero
        loss.backward()

        #SGD or adam
        optimizer.step() #updating the gradients
    
#check accuracy to see if it performs any good
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        
        accuracy = float(num_correct)/float(num_samples)
        

    #model.train()
    return accuracy
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")








