import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total variation loss function                              #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        tv_loss = 0
        diff_h = img[:,:,0:-1,:] - img[:,:,1:,:]
        diff_v = img[:,:,:,0:-1] - img[:,:,:,1:]
        tv_loss += torch.sum(torch.pow(diff_h, 2))
        tv_loss += torch.sum(torch.pow(diff_v, 2))
        tv_loss *= tv_weight
        return tv_loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################