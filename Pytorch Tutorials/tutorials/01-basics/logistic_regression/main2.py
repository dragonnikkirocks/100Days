import torch
import torchvision.transforms as transforms
import torchvision.datasets as dts
from torch.autograd import Variable

traindataset = dts.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
testdataset = dts.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())
batch_size=100
n_iters = 9000
epochs = n_iters / (len(traindataset) / batch_size)
input_dim = 784
output_dim = 10
lr_rate = 0.001




trainloader = torch.utils.data.DataLoader(dataset=traindataset,batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testdataset,batch_size=batch_size,shuffle=False)

class LogisticReg(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LogisticReg,self).__init__()
        self.Linear = torch.nn.Linear(input_dim,output_dim)

    def forward(self,x):
        outputs= self.Linear(x)
        return outputs

model = LogisticReg(input_dim,output_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr_rate)


iteration = 0
for epoch in range(int(epochs)):
    for i,(images,labels ) in enumerate(trainloader):
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        iteration+=1
        if iteration%500==0:

            #calculate accuracy
            correct=0
            total=0
            if iteration%500==0:
            # calculate Accuracy
                with torch.no_grad():
                    for images, labels in testloader:
                        images = Variable(images.view(-1, 28*28))
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total+= labels.size(0)
                        # for gpu, bring the predicted and labels back to cpu fro python operations to work
                        correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))



