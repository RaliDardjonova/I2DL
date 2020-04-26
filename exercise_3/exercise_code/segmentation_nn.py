"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        #self.model = models.vgg16(pretrained=True)
        #self.transform = transforms.Compose([            #[1]
        #transforms.Resize(256),                    #[2]
        #transforms.CenterCrop(224),                #[3]
        #traghnsforms.ToTensor(),                     #[4]
        #transforms.Normalize(                      #[5]
        #mean=[0.485, 0.456, 0.406],                #[6]
        #std=[0.229, 0.224, 0.225]                  #[7]
        #)])
        #self.model = models.resnet101(pretrained=True)
        self.model = models.alexnet(pretrained = True)
        for param in self.model.parameters():
            param.requires_grad = False
        #print(self.model.classifier)
        #self.model.classifier[0] = nn.Conv2d(512, 4096, 7)
        #self.model.classifier[3] = nn.Conv2d(4096, 4096, 1)
        #self.model.classifier[6] = nn.Sequential(nn.Conv2d(4096, num_classes, 1),
        #nn.ConvTranspose2d(num_classes, num_classes, 240, stride=32,
        #                                  bias=False))
        #self.model.fc = nn.Sequential(nn.Conv2d(2048, num_classes, 1),
        #nn.ConvTranspose2d(num_classes, num_classes, 240, stride=32,
        #
        #                                  bias=False))
        self.model.classifier = nn.Sequential(nn.Conv2d(256, num_classes, 1),
        nn.ConvTranspose2d(num_classes, num_classes, 220, stride=4, bias=False))
        #self.fc = nn.Identity()
        #self.model.fc = nn.Conv2d(in_channels=2048,out_channels=num_classes,kernel_size=1,stride=1)
        #self.model.fc['aux'] = nn.Conv2d(in_channels=2048,out_channels=num_classes,kernel_size=1,stride=1)
        print(self.model)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.model.features(x)
        #print(x.shape)
        #x = self.model.forward(x)
        #print(x.shape)
        x = self.model.classifier(x)
        #print(x.shape)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
