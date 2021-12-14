import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W) # batch size, number of color channels, height, width
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        n, _, h_in, w_in = x.shape # batch_size, color channels, height of input volume, width of input wolume
        n_f, n_in, h_f, w_f = self.weight.shape # number of filters, number of in channels, height of filter volume, width of filter volume
        h_out = (h_in - h_f + 2*self.padding) // self.stride + 1 # height of output volume
        w_out = (w_in - w_f + 2*self.padding) // self.stride + 1 # width of output volume
        pad = (self.padding, self.padding)
        x_pad = np.pad(array = x, pad_width = ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), mode = 'constant')
        output_shape = n, n_f, h_out, w_out
        out_weight = np.zeros(output_shape)
        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * self.stride, j * self.stride
                h_end, w_end = h_start + h_f, w_start + w_f
                out_weight[:, :, i, j] = np.sum(x_pad[:, np.newaxis, :, h_start:h_end, w_start:w_end] * self.weight[np.newaxis, :, :, :], axis = (2,3,4))
        out_bias = self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        out = out_weight + out_bias
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        _, c_out, h_out, w_out = dout.shape
        n, _, h_in, w_in = x.shape
        _, _, h_f, w_f = self.weight.shape
        pad = (self.padding, self.padding)
        x_pad = np.pad(array = x, pad_width = ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), mode = 'constant')
        self.db = dout.sum(axis = (0, 2, 3))
        self.dw = np.zeros_like(self.weight)
        self.dx = np.zeros_like(x_pad)
        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * self.stride, j * self.stride
                h_end, w_end = h_start + h_f, w_start + w_f
                self.dx[:, :, h_start:h_end, w_start:w_end] += np.sum(self.weight[np.newaxis, :, :, :] * dout[:, :, i:i+1, j:j+1, np.newaxis], axis = 1)
                self.dw += np.sum(x_pad[:, np.newaxis, :, h_start:h_end, w_start:w_end] * dout[:, :, i:i+1, j:j+1, np.newaxis], axis = 0)
        self.dx = self.dx[:, :, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################