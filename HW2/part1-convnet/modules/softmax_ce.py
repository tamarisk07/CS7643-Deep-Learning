import numpy as np

class SoftmaxCrossEntropy:
    '''
    Compute softmax cross-entropy loss given the raw scores from the network.
    '''
    def __init__(self):
        self.dx = None
        self.cache = None

    def forward(self, x, y):
        '''
        Compute Softmax Cross Entropy Loss
        :param x: raw output of the network: (N, num_classes)
        :param y: labels of samples: (N, )
        :return: computed CE loss of the batch
        '''
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N, _ = x.shape
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        self.cache = (probs, y, N)
        return probs, loss

    def backward(self):
        '''
        Compute backward pass of the loss function
        :return:
        '''
        probs, y, N = self.cache
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        self.dx = dx