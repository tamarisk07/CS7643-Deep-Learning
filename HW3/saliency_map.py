import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, Saliency

from visualizers import SaliencyMap
from data_utils import *
from image_utils import *
from captum_utils import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

X, y, class_names = load_imagenet_val(num=5)
# manually compute saliency maps
sm = SaliencyMap()
sm.show_saliency_maps(X, y, class_names, model)

# ************************************************************************************** #

# use Captum for saliency map

# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
y_tensor = torch.LongTensor(y)

# Example with captum
# Computing and visualizing Integrated Gradient

# int_grads = IntegratedGradients(model)
# attr_ig = compute_attributions(int_grads, X_tensor, target=y_tensor, n_steps=10)
# visualize_attr_maps('visualization/int_grads_captum.png', X, y, class_names, [attr_ig], ['Integrated Gradients'])

##############################################################################
# TODO: Compute/Visualize Saliency using captum.                             #
#       visualize_attr_maps function from captum_utils.py is useful for      #
#       visualizing captum outputs                                           #
#       You can refer to the 'Integrated gradients' visualization            #
#       in the comments above this section as an example                     #
##############################################################################
# Computing saliency maps
sali = Saliency(model)
attr_sali = compute_attributions(sali, X_tensor, target = y_tensor)
visualize_attr_maps('visualization/saliency_captum.png', X, y, class_names, [attr_sali], ['Saliency'])
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
