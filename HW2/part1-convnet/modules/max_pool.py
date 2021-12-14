import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = {}
        self.dx = None

    def _save_mask(self, x, coords):
        mask = np.zeros_like(x)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w)
        idx = np.argmax(x, axis = 2)
        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, c, h*w)[n_idx, c_idx, idx] = 1
        self.cache[coords] = mask

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        n, c, h_in, w_in = x.shape
        h_pool, w_pool = self.kernel_size, self.kernel_size
        h_out = 1 + (h_in - h_pool) // self.stride
        w_out = 1 + (w_in - w_pool) // self.stride
        out = np.zeros((n, c, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                self._save_mask(x_slice, coords = (i, j))
                out[:, :, i, j] = np.max(x_slice, axis = (2, 3))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.dim = (n, c, h_out, w_out)
        self.cache_x = np.array(x, copy=True)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives # n, c, h_out, w_out
        :return:
        '''
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        dout = dout.reshape(self.dim)
        _, _, h_out, w_out = dout.shape
        h_pool, w_pool = self.kernel_size, self.kernel_size
        self.dx = np.zeros_like(self.cache_x)
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                self.dx[:, :, h_start:h_end, w_start:w_end] += dout[:, :, i:i+1, j:j+1]*self.cache[(i,j)]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
