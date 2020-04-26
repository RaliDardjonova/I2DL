import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.relu1 = nn.ELU(inplace = True)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ELU(inplace = True)
        self.maxPool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p = 0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)
        self.relu3 = nn.ELU(inplace = True)
        self.maxPool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p = 0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)
        self.relu4 = nn.ELU(inplace = True)
        self.maxPool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p = 0.4)

        self.dense1 = nn.Linear(6400, 1000)
        self.relu5 = nn.ELU(inplace=True)
        self.dropout5 = nn.Dropout(p = 0.5)

        self.dense2 = nn.Linear(1000,1000)
        self.relu6 = nn.ELU(inplace=True)
        self.dropout6 = nn.Dropout(p = 0.6)
        self.dense3 = nn.Linear(1000,30)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        #x = torch.squeeze(x,0)
        #Sprint(x.shape)
        x = self.maxPool1(F.elu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.maxPool2(F.elu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.maxPool3(F.elu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.maxPool4(F.elu(self.conv4(x)))
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        x = F.elu(self.dense1(x))
        x = self.dropout5(x)
        x = F.elu(self.dense2(x))
        x = self.dropout6(x)
        x = self.dense3(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
