import torch
import torchvision 
from tqdm import tqdm
import matplotlib
import mnist
import torch.nn as nn


class bls(torch.nn.Module):
    def __init__(self, feature_nodes, enhancement_nodes, num_classes):
        super(bls, self).__init__()
        self.fc1 = nn.Linear(784, feature_nodes)
        self.fc2 = nn.Linear(784, feature_nodes)
        self.fc3 = nn.Linear(784, feature_nodes)
        self.fc4 = nn.Linear(784, feature_nodes)
        self.fc5 = nn.Linear(784, feature_nodes)
        self.fc6 = nn.Linear(784, feature_nodes)
        self.fc7 = nn.Linear(784, feature_nodes)
        self.fc8 = nn.Linear(784, feature_nodes)
        self.fc9 = nn.Linear(784, feature_nodes)
        self.fc10 = nn.Linear(784, feature_nodes)

        self.fc31 = nn.Linear(feature_nodes*10, enhancement_nodes)
        self.fc32 = nn.Linear(feature_nodes*10+enhancement_nodes, num_classes)
        # self.fc33 = nn.Linear(6140, 200)

    def forward(self, x):
        B, C, W, H = x.shape
        x = x.squeeze().view(B, -1)
        # print(x.shape)
        feature_nodes1 = torch.sigmoid(self.fc1(x))
        feature_nodes2 = torch.sigmoid(self.fc2(x))
        feature_nodes3 = torch.sigmoid(self.fc3(x))
        feature_nodes4 = torch.sigmoid(self.fc4(x))
        feature_nodes5 = torch.sigmoid(self.fc5(x))
        feature_nodes6 = torch.sigmoid(self.fc6(x))
        feature_nodes7 = torch.sigmoid(self.fc7(x))
        feature_nodes8 = torch.sigmoid(self.fc8(x))
        feature_nodes9 = torch.sigmoid(self.fc9(x))
        feature_nodes10 = torch.sigmoid(self.fc10(x))

        feature_nodes = torch.cat(
            [feature_nodes1, feature_nodes2, feature_nodes3, feature_nodes4, feature_nodes5, feature_nodes6,
             feature_nodes7, feature_nodes8, feature_nodes9, feature_nodes10], 1)
        enhancement_nodes = torch.sigmoid(self.fc31(feature_nodes))
        FeaAndEnhance = torch.cat([feature_nodes, enhancement_nodes], 1)
        outs = self.fc32(FeaAndEnhance)
        # o4 = self.fc32(o4)
        return outs

device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                                # torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
      
BATCH_SIZE = 256
EPOCHS = 30
trainData = mnist.MNIST('./data/',train = True,transform=transform, download = True)

testData = mnist.MNIST('./data/',train = False, transform=transform)


trainDataLoader = torch.utils.data.DataLoader(dataset = trainData,batch_size = BATCH_SIZE,shuffle = True)
testDataLoader = torch.utils.data.DataLoader(dataset = testData,batch_size = BATCH_SIZE)
net = bls(10, 8000, 10)
print(net.to(device))

lossF = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,20], gamma=0.2, last_epoch=-1)
history = {'Test Loss':[],'Test Accuracy':[]}
for epoch in range(1,EPOCHS + 1):
    processBar = tqdm(trainDataLoader,unit = 'step')
    net.train(True)
    for step,(trainImgs,labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs,labels)
        predictions = torch.argmax(outputs, dim = 1)
        accuracy = torch.sum(predictions == labels)/labels.shape[0]
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item()))
        
        if step == len(processBar)-1:
            correct,totalLoss = 0,0
            net.train(False)
            for testImgs,labels in testDataLoader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                loss = lossF(outputs,labels)
                predictions = torch.argmax(outputs,dim = 1)
                
                totalLoss += loss
                correct += torch.sum(predictions == labels)
                
            testAccuracy = correct/(BATCH_SIZE * len(testDataLoader))
            testLoss = totalLoss/len(testDataLoader)
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    processBar.close()

matplotlib.pyplot.plot(history['Test Loss'],label = 'Test Loss')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Loss')
matplotlib.pyplot.show()

matplotlib.pyplot.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Accuracy')
matplotlib.pyplot.show()

torch.save(net,'./model.pth')
