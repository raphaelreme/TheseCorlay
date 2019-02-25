from NeuralNet import NeuralNet
from mxnet import nd
# import numpy as np

def abs(x):
    return nd.abs(x[0])<x[1]

def vague2(x):
    y = x[1]
    x = nd.where(x[0] > 0, x[0], -x[0])
    x = nd.where(x > 1, 2 - x, x)
    return y>x
