import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import GuidedGradCam, GuidedBackprop
from captum.attr import LayerActivation, LayerConductance, LayerGradCam

from data_utils import *
from image_utils import *
from captum_utils import *
import numpy as np

from visualizers import GradCam


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

X, y, class_names = load_imagenet_val(num=5)

# FOR THIS SECTION ONLY, we need to use gradients. We introduce a new model we will use explicitly for GradCAM for this.
gc_model = torchvision.models.squeezenet1_1(pretrained=True)
gc = GradCam()

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)

# Guided Back-Propagation
gbp_result = gc.guided_backprop(X_tensor,y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gbp_result.shape[0]):
    plt.subplot(1, 5, i + 1)
    img = gbp_result[i]
    img = rescale(img)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/guided_backprop.png')



# GradCam
# GradCAM. We have given you which module(=layer) that we need to capture gradients from, which you can see in conv_module variable below
gc_model = torchvision.models.squeezenet1_1(pretrained=True)
for param in gc_model.parameters():
    param.requires_grad = True

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)
gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gradcam_result.shape[0]):
    gradcam_val = gradcam_result[i]
    img = X[i] + (matplotlib.cm.jet(gradcam_val)[:,:,:3]*255)
    img = img / np.max(img)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/gradcam.png')


# As a final step, we can combine GradCam and Guided Backprop to get Guided GradCam.
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)
gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)
gbp_result = gc.guided_backprop(X_tensor, y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gradcam_result.shape[0]):
    gbp_val = gbp_result[i]
    gradcam_val = np.expand_dims(gradcam_result[i], axis=2)

    # Pointwise multiplication and normalization of the gradcam and guided backprop results (2 lines)
    img = gradcam_val * gbp_val

    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img = np.float32(img)
    img = torch.from_numpy(img)
    img = deprocess(img)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/guided_gradcam.png')


# **************************************************************************************** #
# Captum
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
y_tensor = torch.LongTensor(y)

conv_module = model.features[12]

##############################################################################
# TODO: Compute/Visualize GuidedBackprop and Guided GradCAM as well.         #
#       visualize_attr_maps function from captum_utils.py is useful for      #
#       visualizing captum outputs                                           #
#       Use conv_module as the convolution layer for gradcam                 #
##############################################################################
ggc_grads = GuidedGradCam(model,conv_module)
attr_ggc = compute_attributions(ggc_grads, X_tensor, target=y_tensor)

gbp_grads = GuidedBackprop(model)
attr_gbp = compute_attributions(gbp_grads, X_tensor, target=y_tensor)

visualize_attr_maps('visualization/backprop_gradcam_captum.png', X, y, class_names, [attr_ggc, attr_gbp], ['Guided GradCAM', 'Guided Backprop'])

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

# Try out different layers and see observe how the attributions change

layer = model.features[3]

# Example visualization for using layer visualizations 
# layer_act = LayerActivation(model, layer)
# layer_act_attr = compute_attributions(layer_act, X_tensor)
# layer_act_attr_sum = layer_act_attr.mean(axis=1, keepdim=True)


##############################################################################
# TODO: Visualize Individual Layer Gradcam and Layer Conductance (similar    #
# to what we did for the other captum sections, using our helper methods),   #
# but with some preprocessing calculations.                                  #
#                                                                            #
# You can refer to the LayerActivation example above and you should be       #
# using 'layer' given above for this section                                 #
#                                                                            #
# Also note that, you would need to customize your 'attr_preprocess'         #
# parameter that you send along to 'visualize_attr_maps' as the default      #
# 'attr_preprocess' is written to only to handle multi channel attributions. #
#                                                                            #
# For layer gradcam look at the usage of the parameter relu_attributions     #
##############################################################################
# Layer gradcam aggregates across all channels
con = LayerConductance(model, layer)
attr_con = compute_attributions(con, X_tensor, target = y_tensor)
attr_con = torch.mean(attr_con, axis = 1, keepdim = True).permute(0, 2, 3, 1)

gc = LayerGradCam(model, layer)
attr_gc = compute_attributions(gc, X_tensor, target = y_tensor)
attr_gc = torch.mean(attr_gc, axis = 1, keepdim = True).permute(0, 2, 3, 1)

visualize_attr_maps('visualization/layer_conductance_gradcam_captum.png', X, y, class_names, [attr_con, attr_gc],['Layer Conductance', 'Layer GradCam'], 
                    attr_preprocess = lambda attr: attr.detach().numpy())
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

