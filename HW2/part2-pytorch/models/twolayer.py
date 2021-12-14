import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size, bias = True)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes, bias = True)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        n,c,h_in,w_in = x.shape
        x_in = torch.reshape(x, (n, -1))
        fc1_out = self.fc1(x_in)
        sig_out = self.sig(fc1_out)
        out = self.fc2(sig_out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out