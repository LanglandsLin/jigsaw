import torch.nn as nn
import torch
from torchvision import models
from net.st_gcn import Model as GCN
class AlexNet(nn.Module):
    def __init__(self, model_type):
        super(AlexNet, self).__init__()
        self.model_type = model_type
        if model_type == 'pre':
            model = models.alexnet(pretrained=True)
            self.features = model.features
            fc = nn.Linear(256, 512)
            #fc.bias = model.classifier[1].bias
            #fc.weight = model.classifier[1].weight

            self.classifier = nn.Sequential(
                    #nn.Dropout(),
                    fc)
                    #nn.ReLU(inplace=True))  

        if model_type == 'new':
            self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 11, 4, 2),
                    nn.ReLU(inplace = True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(64, 192, 5, 1, 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(192, 384, 3, 1, 1),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(384, 256, 3, 1, 1),
                    nn.ReLU(inplace=True))
                    #nn.MaxPool2d(3, 2, 0))
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True))
            
    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        out  = self.classifier(x)
        return out


class O3N(nn.Module):
    def __init__(self, model_type, output_size):
        super(O3N, self).__init__()
        #self.alexnet = AlexNet(model_type)
        self.gcn = GCN(3, output_size, {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, True)
        #self.classifier = nn.Sequential(
                    #nn.Conv2d(256, output_size, kernel_size=1)
                    #nn.Dropout(),
                    #nn.Linear(750, 4096),
                    #nn.ReLU(inplace=True),
                    #nn.Dropout(),
                    #nn.Linear(4096, output_size))
                    #nn.ReLU(inplace=True))


    def forward(self, input):
        #shape = input.shape
        #input = input.view(*shape[0:3], shape[3] * shape[4])
        X = self.gcn(input)
        #X = X.mean(dim=1).view(shape[0], -1)
        #X = self.classifier(X)
        return X
