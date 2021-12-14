from ._base_optimizer import _BaseOptimizer
import numpy as np
class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :param gradient: The Gradient computed in forward step
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)
        #############################################################################
        # TODO:                                                                     #
        #    1) Update model weights based on the learning rate and gradients       #
        #############################################################################
        if 'W1' in model.weights:
            model.weights["W1"] = model.weights["W1"] - self.learning_rate*model.gradients["W1"]
        if 'W2' in model.weights:
            model.weights["W2"] = model.weights["W2"] - self.learning_rate*model.gradients["W2"]
        if 'b1' in model.weights:
            model.weights["b1"] = model.weights["b1"] - self.learning_rate*model.gradients["b1"]
        if 'b2' in model.weights:
            model.weights["b2"] = model.weights["b2"] - self.learning_rate*model.gradients["b2"]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
