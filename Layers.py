# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd



class Layers:
    """One layers of a neural net. (No computation on GPU. Nothing is really optimized in this class)

    Has a n neurones (size of the output) and the p inputs.
    There is therefore n(p+1) parameters (weigths + bias).

    One can precise each activation function for each neurone.
    One can also fixed a set of connections for the learning algorithm."""

    def __init__(self, output_size, input_size, function = nd.sigmoid, fixed = False, batch_size = 1):
        if output_size < 1 or input_size < 1 or batch_size < 1:
            raise ValueError("Sizes must be positive integer")
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size

        self.weights = [[nd.random.normal(loc = 0, scale = 1, shape = 1) for j in range(input_size)] for i in range(output_size)]
        self.bias = [nd.random.normal(loc = 0, scale = 1, shape = 1) for i in range(output_size)]

        ###############################
        ###### Compute functions ######
        self.set_functions(function)


        ###########################################
        ###### Compute the fixed parameters #######
        self.set_fixed(fixed)



    def set_fixed(self, fixed):
        """Fixed has to be a boolean or a ndarray. The sizes accepted are n * (p+1), n and p+1. The non given value are deduced.
        If n values are given, they apply for each neurones. (If one neurones is frozen, all its connections too).
        If p + 1 values are given, they apply for each input. (If an input is frozen, all its connection too).
        If one value is given (Booleean). Then all the connection will follow its value."""

        message = "'fixed' argument can either be one boolean for all the parameters or n booleans, one for each neurone and their connections, or p+1 booleans, one for each input (+bias, give the bias first) and their connections or a matrix of n*(p+1) booleans for each connection"
        try:
            if len(fixed.shape) == 1:
                if fixed.shape[0] == self.input_size + 1:
                    self.bias_fixed = [bool(fixed[0]) for i in range(self.output_size)]
                    self.weights_fixed = [[bool(fixed[1+j]) for j in range(self.input_size)] for i in range(self.output_size)]
                elif fixed.shape[0] == self.output_size:
                    self.bias_fixed = [bool(fixed[i]) for i in range(self.output_size)]
                    self.weights_fixed = [[bool(fixed[i]) for j in range(self.input_size)] for i in range(self.output_size)]
                else:
                    raise ValueError(message)
            elif fixed.shape == (self.output_size, self.input_size + 1):
                self.bias_fixed = [bool(fixed[i,0]) for i in range(self.output_size)]
                self.weights_fixed = [[bool(fixed[i,1+j]) for j in range(self.input_size)] for i in range(self.output_size)]
            else:
                raise ValueError(message)
        except AttributeError:
            self.weights_fixed = [[bool(fixed) for j in range(self.input_size)] for i in range(self.output_size)]
            self.bias_fixed = [bool(fixed) for i in range(self.output_size)]


    def set_functions(self, function):
        try:
            if len(function) != self.output_size: # If len is defined, we test that it's a good one, otherwise there is a TypeError and we deal with it in the except bloc
                raise ValueError("'function' argument can either be one function for all the neurones or n functions (one for each)")
            self.functions = function
        except TypeError:
            self.functions = [function for i in range(self.output_size)]

    def attach_grad(self):
        for i in range(self.output_size):
            if not self.bias_fixed[i]:
                self.bias[i].attach_grad()
            else:
                self.bias[i].attach_grad(grad_req = 'null')
            for j in range(self.input_size):
                if not self.weights_fixed[i][j]:
                    self.weights[i][j].attach_grad()
                else:
                    self.weights[i][j].attach_grad(grad_req = 'null')

    def compute(self, input):
        """Return the batch_size outputs given the batch_size inputs."""
        Z = []

        for k in range(self.batch_size):
            Z.append([])
            for i in range(self.output_size):
                Z[k].append(self.bias[i])
                for j in range(self.input_size):
                    Z[k][i] += input[k][j]*self.weights[i][j]


        for k in range(self.batch_size):
            for i in range(self.output_size):
                Z[k][i] = self.functions[i](Z[k][i])

        return Z
