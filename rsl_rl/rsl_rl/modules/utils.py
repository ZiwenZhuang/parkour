import torch
import torch.nn as nn

def get_activation_Cls(activation_name):
    if hasattr(nn, activation_name):
        return getattr(nn, activation_name)
    
    if activation_name == "elu":
        return nn.ELU
    elif activation_name == "selu":
        return nn.SELU
    elif activation_name == "relu":
        return nn.ReLU
    elif activation_name == "crelu":
        return nn.ReLU
    elif activation_name == "lrelu":
        return nn.LeakyReLU
    elif activation_name == "tanh":
        return nn.Tanh
    elif activation_name == "sigmoid":
        return nn.Sigmoid
    else:
        print("invalid activation function!")
        return None

def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w
