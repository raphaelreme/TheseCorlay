# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd



class Layers:
    """One layers of a neural net.

    Has a n neurones (size of the output) and the p inputs.
    There is therefore n(p+1) parameters (weigths + bias).

    One can precise each activation function for each neurone.
    One can also fixed a set of connections for the learning algorithm."""

    def __init__(self, output_size, input_size, function = nd.sigmoid, ctx = mx.cpu(0), fixed = False):
        if output_size < 1 and input_size < 1:
            raise ValueError("Sizes must be positive integer")

        self.output_size = output_size
        self.input_size = input_size

        self.ctx = ctx

        self.weights = nd.random.normal(loc = 0, scale = 1, shape = (output_size, input_size), ctx = self.ctx)
        self.bias = nd.random.normal(loc = 0, scale = 1, shape = output_size, ctx = self.ctx)

        ###############################
        ###### Compute functions ######
        try:
            if len(function) != output_size: # If len is defined, we test that it's a good one, otherwise there is a TypeError and we deal with it in the except bloc
                raise ValueError("'function' argument can either be one function for all the neurones or n functions (one for each)")
            self.function_is_one = False
            self.function = function
        except TypeError:
            self.function_is_one = True
            self.function = function


        ###########################################
        ###### Compute the fixed parameters #######
        self.set_fixed(fixed)



    def set_fixed(self, fixed):
        message = "'fixed' argument can either be one boolean for all the parameters or n booleans, one for each neurone and their connections, or p+1 booleans, one for each input (+bias, give the bias first) and their connections or a matrix of n*(p+1) booleans for each connection"
        try:
            if len(fixed.shape) == 1:
                if fixed.shape[0] == self.input_size + 1:
                    self.bias_fixed = nd.full(self.bias.shape, fixed[0], ctx = self.ctx)
                    self.weights_fixed = nd.dot(nd.full((self.output_size, 1), 1), fixed[1:].reshape((1, self.input_size))).copyto(self.ctx)
                elif fixed.shape[0] == self.output_size:
                    self.bias_fixed = fixed.copyto(self.ctx)
                    self.weights_fixed = nd.dot(fixed.reshape((self.output_size, 1)), nd.full((1, self.input_size), 1)).copyto(self.ctx)
                else:
                    raise ValueError(message)
            elif fixed.shape == (self.output_size, self.input_size + 1):
                self.bias_fixed = fixed[:,0].copyto(self.ctx)
                self.weights_fixed = fixed[:,1:].copyto(self.ctx)
            else:
                raise ValueError(message)
        except AttributeError:
            self.weights_fixed = nd.full(self.weights.shape, fixed, ctx = self.ctx)
            self.bias_fixed = nd.full(self.bias.shape, fixed, ctx = self.ctx)

        self.bias_fixed = self.bias_fixed.astype('int8')
        self.weights_fixed = self.weights_fixed.astype('int8')






    def compute(self, input):
        Z = nd.dot(self.weights, input) + self.bias

        if self.function_is_one:
            return self.function(Z)
        else:
            A = Z.zeros_like()
            for i in range(len(Z)):
                for j in range(len(Z[0])): #If input is of shape p * batch_size or of size p it still work
                    A[i][j] = self.function[i](Z[i][j])
