import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from image_utils import preprocess, deprocess, SQUEEZENET_MEAN, SQUEEZENET_STD

from scipy.ndimage.filters import gaussian_filter1d

import random

class ClassVisualization:
    def jitter(self, X, ox, oy):
        """
        Helper function to randomly jitter an image.

        Inputs
        - X: PyTorch Tensor of shape (N, C, H, W)
        - ox, oy: Integers giving number of pixels to jitter along W and H axes

        Returns: A new PyTorch Tensor of shape (N, C, H, W)
        """
        if ox != 0:
            left = X[:, :, :, :-ox]
            right = X[:, :, :, -ox:]
            X = torch.cat([right, left], dim=3)
        if oy != 0:
            top = X[:, :, :-oy]
            bottom = X[:, :, -oy:]
            X = torch.cat([bottom, top], dim=2)
        return X

    def blur_image(self, X, sigma=1):
        X_np = X.cpu().clone().numpy()
        X_np = gaussian_filter1d(X_np, sigma, axis=2)
        X_np = gaussian_filter1d(X_np, sigma, axis=3)
        X.copy_(torch.Tensor(X_np).type_as(X))
        return X

    def create_class_visualization(self, target_y, class_names, model, dtype, **kwargs):
        """
        Generate an image to maximize the score of target_y under a pretrained model.

        Inputs:
        - target_y: Integer in the range [0, 1000) giving the index of the class
        - model: A pretrained CNN that will be used to generate the image
        - dtype: Torch datatype to use for computations

        Keyword arguments:
        - l2_reg: Strength of L2 regularization on the image
        - learning_rate: How big of a step to take
        - num_iterations: How many iterations to use
        - blur_every: How often to blur the image as an implicit regularizer
        - max_jitter: How much to gjitter the image as an implicit regularizer
        - show_every: How often to show the intermediate result
        """

        model.eval()

        model.type(dtype)
        l2_reg = kwargs.pop('l2_reg', 1e-3)
        learning_rate = kwargs.pop('learning_rate', 25)
        num_iterations = kwargs.pop('num_iterations', 100)
        blur_every = kwargs.pop('blur_every', 10)
        max_jitter = kwargs.pop('max_jitter', 16)
        show_every = kwargs.pop('show_every', 25)

        # Randomly initialize the image as a PyTorch Tensor, and also wrap it in
        # a PyTorch Variable.
        img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype)
        img_var = Variable(img, requires_grad=True)

        for t in range(num_iterations):
            # Randomly jitter the image a bit; this gives slightly nicer results
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.copy_(self.jitter(img, ox, oy))

            ########################################################################
            # TODO: Use the model to compute the gradient of the score for the     #
            # class target_y with respect to the pixels of the image, and make a   #
            # gradient step on the image using the learning rate. Don't forget the #
            # L2 regularization term!                                              #
            # Be very careful about the signs of elements in your code.            #
            ########################################################################
            pred = model(img_var)
            pred[0, target_y].backward()
            img_var.grad -= 2 * l2_reg * img_var
            dx = learning_rate * img_var.grad / torch.norm(img_var.grad)
            img_var.data.add_(dx)
            img_var.grad.fill_(0)
            ########################################################################
            #                             END OF YOUR CODE                         #
            ########################################################################

            # Undo the random jitter
            img.copy_(self.jitter(img, -ox, -oy))

            # As regularizer, clamp and periodically blur the image
            for c in range(3):
                lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
                hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
                img[:, c].clamp_(min=lo, max=hi)
            if t % blur_every == 0:
                self.blur_image(img, sigma=0.5)

            # Periodically show the image
            if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
                plt.imshow(deprocess(img.clone().cpu()))
                class_name = class_names[target_y]
                plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
                plt.gcf().set_size_inches(4, 4)
                plt.axis('off')
                plt.savefig('visualization/class_visualization_iter_{}'.format(t+1))
        return deprocess(img.cpu())
