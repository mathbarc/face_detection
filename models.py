## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class FaceDetectionNet(nn.Module):

    def __init__(self):
        super(FaceDetectionNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.batchNorm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.batchNorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.batchNorm3 = nn.BatchNorm2d(16)
        
        self.flatten = nn.Flatten()
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dense1 = nn.Linear(18528, 256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(256, 136)
        
        #self.output = nn.Tanh()
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

                
        x = F.relu(self.conv1(x))        
        x = self.pool1(x)
        
        x = self.batchNorm1(x)
        
        x = F.relu(self.conv2(x))
        i = self.pool2(x)
        
        
        i = self.batchNorm2(i)
        
        
        x = F.relu(self.conv3(i))
        x = self.pool3(x)  

        x = self.batchNorm3(x)

        i = self.flatten(i)
        x = self.flatten(x)
        
        #x = x.view(x.size(0), -1)
        #i = i.view(i.size(0), -1)
        
        #print(x.size())
        #print(i.size())
        
        x = torch.cat((x,i),1)
        x = self.dropout1(x)
        
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        # x = self.output(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
