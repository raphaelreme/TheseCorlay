# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd

from NeuralNet import NeuralNet
from Layer import Layer

class FoldingNet(NeuralNet):

    def __init__(self, symetries, ctx = mx.cpu(0), optimize = False):
        self.layers = []
        self.sizes = []
        for n,b in symetries:
            self.generate_layers(n, b, ctx, optimize)
        self.layersNumber = len(self.layers)



    def generate_layers(self, n, b, ctx = mx.cpu(0), optimize = False):
        W = nd.eye(len(n))
        for i in range(1, len(n)):
            if n[i] == 0:
                continue
            rot_minus_theta = nd.eye(len(n))
            n_temp = nd.dot(W, n)
            theta = nd.arctan(n_temp[i]/n_temp[0])
            c = nd.cos(theta)
            s = nd.sin(theta)
            rot_minus_theta[0,0] = c
            rot_minus_theta[i,i] = c
            rot_minus_theta[0,i] = s
            rot_minus_theta[i,0] = -s
            W = nd.dot(rot_minus_theta, W)
        L = Layer(len(n), len(n), function = nd.identity, fixed = True, ctx = ctx)
        L.bias = nd.zeros(len(n))
        L.weights = W
        self.layers.append(L)
        self.sizes.append(len(n))
        self.sizes.append(len(n))

        # Symetry for the plane x = -b
        # n1' = |n1 + b| - b. ni' = ni.
        function = [nd.relu]*2 + [nd.identity] * (len(n)-1)
        L = Layer(len(n)+1, len(n), function = function, fixed = True, ctx = ctx)
        L.bias = nd.zeros(len(n)+1)
        L.bias[0] = -b
        L.bias[1] = b
        L.weights = nd.eye(len(n)+1, len(n), -1)
        L.weights[0,0] = -1
        self.layers.append(L)
        self.sizes.append(len(n)+1)

        L = Layer(len(n), len(n) + 1, function = nd.identity, fixed = True, ctx = ctx)
        L.bias = nd.zeros(len(n))
        L.bias[0] = -b
        L.weights = nd.eye(len(n), len(n)+1, 1)
        L.weights[0,0] = 1
        self.layers.append(L)
        self.sizes.append(len(n))

        L = Layer(len(n), len(n), function = nd.identity, fixed = True, ctx = ctx)
        L.bias = nd.zeros(len(n))
        L.weights = W.T
        self.layers.append(L)
        self.sizes.append(len(n))
