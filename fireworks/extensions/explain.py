import torch
import copy
import numpy as np
from itertools import count, product
import fireworks
import logging

logger = logging.getLogger(__name__)

def GradientxInput(model, input, input_column=None, output_column=None):
    """
    Returns attributions for a model against an input using the Gradient x Input method.
    This method computes D(X)*J(Y), where D(X) is a diagonalized matrix representation of input X
    and Y = model(X).

    The first dimension of input and output is assumed to be the batch dimension.
    """

    if input_column:
        X = input[input_column]
    else:
        X = input

    # Compute Jacobian
    X.requires_grad=True
    output = model(input)
    if output_column:
        Y = output[output_column]
    else:
        Y = output
    
    batch_size = X.shape[0]
    out_shape = Y.shape[1:]
    out_indices = [range(r) for r in out_shape]
    in_shape = X.shape[1:]
    gxi = torch.zeros(batch_size, *out_shape, *in_shape)
    for y, row in zip(Y, count()):
        for multiindex in product(*out_indices):
            e = y[multiindex]
            e.backward(retain_graph=True)
            windowed_grad = X.grad.data[row]
            windowed_X = X.data[row]
            gxi[row][multiindex] = (windowed_X * windowed_grad).clone()
            X.grad.data.zero_()
        
    return gxi

def IntegratedGradients(model, input, input_column=None, output_column=None, baseline=0, steps=100):
    """
    Returns attributions for a model against an input using the Integrated Gradients method which
    is described in Sundararajan et al. (http://arxiv.org/abs/1703.01365)

    The first dimension of input and output is assumed to be the batch dimension.
    """
    if input_column:
        X = input[input_column]
    else:
        X = input
    output = model(input)
    if output_column:
        Y = output[output_column]
    else:
        Y = output
    attributions = None
    batch_size = X.shape[0]
    out_shape = Y.shape[1:]
    out_indices = [range(r) for r in out_shape]
    in_shape = X.shape[1:]
    # alpha parameterizes Xs from baseline to its original value
    for alpha in np.linspace(0., 1.0, steps):
        # Get a new output Ys for each Xs
        Xs = baseline + (X-baseline)*alpha 
        Xs.requires_grad=True
        if input_column:
            input_s = fireworks.Message()
            for column in input.columns:
                try:
                    input_s[column] = copy.deepcopy(input[column])
                except RuntimeError:
                    pass
            input_s[input_column] = Xs
        else:
            input_s = Xs
        output = model(input_s)
        if output_column:
            Ys = output[output_column]
        else:
            Ys = output
        # We compute gradients attributions for each Xs
        gradient = torch.zeros(batch_size, *out_shape, *in_shape)
        for y, row in zip(Ys, count()):
            for multiindex in product(*out_indices):
                e = y[multiindex]
                e.backward(retain_graph=True)
                windowed_grad = Xs.grad.data[row]
                gradient[row][multiindex] = (windowed_grad).clone()
                Xs.grad.data.zero_()        
        # Update attributions
        if attributions is None:
            attributions = gradient
        else:
            attributions += gradient
    # Compute discrete integral and multiply by shifted input
    attributions = attributions / steps 
    if Ys.device.type == 'cuda':
        attributions = attributions.cuda(device=Ys.device.index)
    for row in  range(batch_size):
        windowed_X = X.data[row] - baseline
        for multiindex in product(*out_indices):
            attributions[row][multiindex] = (windowed_X * attributions[row][multiindex]).clone()

    return attributions

def e_LRP(model, input, input_column=None, output_column=None):
    """
    Returns attributions for a model against an input using the epsilon-LRP (Layer-wise 
    relevance propagation) method which is described in Zeiler et al. 
    (https://arxiv.org/abs/1311.2901)

    The first dimension of input and output is assumed to be the batch dimension.
    """
    pass 

def DeepLift(model, input, input_column=None, output_column=None):
    """
    Returns attributions for a model against an input using the DeepLift method which
    is described in Shrikumar et al. (https://arxiv.org/abs/1704.02685)

    The first dimension of input and output is assumed to be the batch dimension.
    """
    pass 

def Occlusion_1(model, input, input_column=None, output_column=None, baseline = 0):
    """
    Returns attributions for a model against an input using the Occlusion-1 method which
    is described in Bach et al. (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)

    The first dimension of input and output is assumed to be the batch dimension.
    """
    if input_column:
        X = input[input_column]
    else:
        X = input
    output = model(input)
    if output_column:
        Y = output[output_column]
    else:
        Y = output

    batch_size = X.shape[0]
    in_shape = X.shape[1:]
    in_indices = [range(r) for r in in_shape]
    out_shape = Y.shape[1:]
    out_indices = [range(r) for r in out_shape]

    occlusions = torch.zeros(batch_size, *out_shape, *in_shape)

    for multiindex in product(*in_indices):
        # Occlude the corresponding element of X 
        X_occluded = copy.deepcopy(X)
        X_occluded[:, multiindex] = baseline
        if input_column:
            input_occluded = copy.deepcopy(input)
            input_occluded[input_column] = X_occluded
        else:
            input_occluded = X_occluded
        output = model(input_occluded)
        if output_column:
            Y_occluded = output[output_column]
        else:
            Y_occluded = output
        occlusion = (Y - Y_occluded)
        if out_shape:
            occlusions[:, :, multiindex] = occlusion.reshape(batch_size, *out_shape, 1)
        else:
            occlusions[:, multiindex] = occlusion.reshape(batch_size, 1)
            
    return occlusions