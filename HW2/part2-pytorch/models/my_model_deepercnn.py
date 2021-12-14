import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(128)
        self.batch3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)
        self.dropout1 = nn.Dropout(p = 0.05)
        self.dropout2 = nn.Dropout(p = 0.1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x_conv_relu1 = F.relu(self.batch1(self.conv1(x))) # 10,32,32,32
        x_conv_relu2 = F.relu(self.conv2(x_conv_relu1)) #10,64,32,32
        x_pool2 = F.max_pool2d(x_conv_relu2, kernel_size = 2, stride = 2) #10,64,16,16
        x_conv_relu3 = F.relu(self.batch2(self.conv3(x_pool2))) #10,128,16,16
        x_conv_relu4 = F.relu(self.conv4(x_conv_relu3)) # 10,128,16,16
        x_pool4 = F.max_pool2d(x_conv_relu4, kernel_size = 2, stride = 2) #10,128,8,8
        x_drop1 = self.dropout1(x_pool4) #10,128,8,8
        x_conv_relu5 = F.relu(self.batch3(self.conv5(x_drop1))) #10,256,8,8
        x_conv_relu6 = F.relu(self.conv6(x_conv_relu5)) #10,256,8,8
        x_pool6 = F.max_pool2d(x_conv_relu6, kernel_size = 2, stride = 2) #10,256,4,4
        x_trans = x_pool6.view(-1, 4096)
        x_fcdrop1 = self.dropout2(x_trans) #10,4096
        x_fc1 = F.relu(self.fc1(x_fcdrop1)) #10,2048
        x_fc2 = F.relu(self.fc2(x_fc1)) #10,1024
        x_fcdrop2 = self.dropout2(x_fc2)
        x_fc3 = self.fc3(x_fcdrop2) #10,512
        x_fc4 = self.fc4(x_fc3) #10,10
        outs = x_fc4
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs