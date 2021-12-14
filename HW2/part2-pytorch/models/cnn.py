import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.convs = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride = 1, padding = 0)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.fc = nn.Linear(5408,10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        conv_out = self.convs(x)
        relu_out = self.act(conv_out)
        pool_out = self.pool(relu_out)
        fc_in = pool_out.view(-1, 5408)
        outs = self.fc(fc_in)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs