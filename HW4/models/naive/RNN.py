import numpy as np
import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """
        You will need to complete the class init function, forward function and hidden layer initialization.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
                output_size (int): the size of the output layer

            Returns:
                None
        """
        super(VanillaRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #############################################################################
        # TODO:                                                                     #
        #    Initialize parameters and layers as you wish. You should to            #
        #    include a hidden unit, an output unit, a tanh function for the hidden  #
        #    unit.                                                                  #
        #    You MUST NOT use Pytorch RNN layers(nn.RNN, nn.LSTM, etc).             #
        #############################################################################
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size)
        self.W_ih = np.random.randn(self.hidden_size, self.input_size)
        self.W_ho = np.random.randn(self.output_size, self.hidden_size)
        self.b_h = np.random.randn(self.hidden_size, 1)
        self.b_o = np.random.randn(self.output_size, 1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x, hidden):
        """ The forward function of the Vanilla RNN
            Args:
                x (tensor): a batch of data of shape (batch_size, input_size) at one time step
                hidden (tensor): the hidden value of previous time step of shape (batch_size, hidden_size)

            Returns:
                output (FloatTensor): the output tensor of shape (output_size, batch_size)
                hidden (FloatTensor): the hidden value of current time step of shape (hidden_size, batch_size)
        """


        output = None

        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass for the Vanilla RNN. Note that we are only   #
        #   going over one time step. Please refer to the structure in the notebook.##
        #############################################################################
        X_ih = np.dot(self.W_ih, x) # hidden_size, batch_size
        U_ih = np.dot(self.W_hh, hidden) # hidden_size, batch_size
        hidden = np.tanh(U_ih + X_ih + self.b_h)
        output = np.dot(self.W_ho, hidden) + self.b_o # output_size, batch_size
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return  torch.FloatTensor(output), torch.FloatTensor(hidden)
