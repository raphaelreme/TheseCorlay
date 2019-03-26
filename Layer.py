# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd

import activation



class Layer:
    """One layer of a neural net.

    Has a output_size neurones and input_size inputs.
    There is therefore output_size x (input_size+1) parameters (weights + bias).

    One can precise each activation function for each neurone.
    One can also fixed a set of connections for the learning algorithm.
    If no weights/bias are given they are choosen randomly."""


    VERSION = 1.3
    decoding_class = "Layer"

    def __init__(self, output_size, input_size, weights = None, bias = None, function = nd.sigmoid, ctx = mx.cpu(0), fixed = False):
        if output_size < 1 and input_size < 1:
            raise ValueError("Sizes must be positive integer")

        self.output_size = output_size
        self.input_size = input_size
        self.total_size = output_size * (input_size + 1) #Number of parameters

        self.ctx = ctx

        ######################################
        ###### Set all the parameters ########
        self.set_bias(bias)
        self.set_weights(weights)
        self.set_function(function)
        self.set_fixed(fixed)

        ######################################
        ###### Prepare gradient descent ######
        self.init_grad()

    def set_bias(self, bias = None):
        """Set the bias, if none is given, set them randomly."""

        if bias is not None:
            assert bias.shape == (self.output_size,), "Wrong shape : Should be (output_size,)."
            self.bias = bias.copyto(self.ctx)
        else:
            self.bias = nd.random.normal(loc = 0, scale = 1, shape = self.output_size, ctx = self.ctx)

    def set_weights(self, weights = None):
        """Set the weights, if none si given then set them randomly."""

        if weights is not None:
            assert weights.shape == (self.output_size, self.input_size), "Wrong shape : Should be (output_size, input_size)."
            self.weights = weights.copyto(self.ctx)
        else:
            self.weights = nd.random.normal(loc = 0, scale = 1, shape = (self.output_size, self.input_size), ctx = self.ctx)


    def set_function(self, function):
        try:
            if len(function) != self.output_size:
                raise ValueError("'function' argument can either be one function for all the neurones or n functions (one for each)")
            self.function_is_one = False
            self.function = function
        except TypeError:
            self.function_is_one = True
            self.function = function


    def set_fixed(self, fixed):
        """Fixed has to be a boolean or a ndarray. The sizes accepted are n * (p+1), n and p+1. The non given value are deduced.
        If n values are given, they apply for each neurones. (If one neurones is fixed, all its connections too).
        If p + 1 values are given, they apply for each input. (If an input is pixed, all its connections too).
        If a boolean is given, then all the connections will follow its value.

        If an uncorrect argument is given then :
        If it has not shape as an attribute, it will cast it in boolean.
        Otherwise if the sizes in shape doesn't fit, it will raise a ValueError.
        BUT (CAREFUL) if the shape is correct, the values aren't tested."""

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
            self.weights_fixed = nd.full(self.weights.shape, bool(fixed), ctx = self.ctx)
            self.bias_fixed = nd.full(self.bias.shape, bool(fixed), ctx = self.ctx)

        self.bias_fixed = self.bias_fixed.astype('uint8')
        self.weights_fixed = self.weights_fixed.astype('uint8')

    def init_grad(self):
        self.bias.attach_grad()
        self.weights.attach_grad()

        self.t = 1

        self.v_bias = self.bias.zeros_like()
        self.sqr_bias = self.bias.zeros_like()
        self.v_weights = self.weights.zeros_like()
        self.sqr_weights = self.weights.zeros_like()


    def adam_descent(self, batch_size = 32, lr = 0.001):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8

        g = self.bias.grad/batch_size
        self.v_bias = beta1 * self.v_bias + (1. - beta1) * g
        self.sqr_bias = beta2 * self.sqr_bias + (1. - beta2) * nd.square(g)
        v_bias_corr = self.v_bias/(1. - beta1**self.t)
        sqr_bias_corr = self.sqr_bias/(1. - beta2**self.t)
        div = lr * v_bias_corr/(nd.sqrt(sqr_bias_corr) + eps_stable)

        self.bias[:] = self.bias - (1 - self.bias_fixed).astype("float32") * div
        #self.bias = nd.where(self.bias > maximum, maximum, self.bias)
        #self.bias = nd.where(self.bias < -maximum, -maximum, self.bias)

        g = self.weights.grad/batch_size
        self.v_weights = beta1 * self.v_weights + (1. - beta1) * g
        self.sqr_weights = beta2 * self.sqr_weights + (1. - beta2) * nd.square(g)
        v_bias_corr = self.v_weights/(1. - beta1**self.t)
        sqr_bias_corr = self.sqr_weights/(1. - beta2**self.t)
        div = lr * v_bias_corr/(nd.sqrt(sqr_bias_corr) + eps_stable)

        self.weights[:] = self.weights - (1-self.weights_fixed).astype("float32") * div

        self.t += 1


    def compute(self, input):
        """Compute the output of the layers given the input.
        The input has to be a ndarray of shape p or (p, batch_size)."""
        Z = (nd.dot(self.weights, input).T + self.bias).T

        if self.function_is_one:
            return self.function(Z)
        else:
            A = Z.zeros_like()
            for i in range(self.output_size):
                A[i] = self.function[i](Z[i])
            return A

    def to_string(self):
        s = "//" + type(self).decoding_class + " V" + str(type(self).VERSION) + "\n\n"

        s += "//Sizes :\n"
        s += str(self.output_size) + " " + str(self.input_size) + "\n\n"

        s += "//Bias :\n"
        for b in self.bias:
            s += str(b.asscalar()) + " "
        s += "\n\n"

        s += "//Weights :\n"
        for line in self.weights:
            for w in line:
                s += str(w.asscalar()) + " "
            s += "\n"
        s += "\n"

        s += "//Functions :\n"
        if self.function_is_one:
            s += self.function.__name__ + "\n\n"
        else:
            for function in self.function:
                s += function.__name__ + " "
            s += "\n\n"

        s += "//Fixed bias : \n"
        for b in self.bias_fixed:
            s += str(b.asscalar()) + " "
        s += "\n\n"

        s+= "//Fixed weights :\n"
        for line in self.weights_fixed:
            for w in line:
                s += str(w.asscalar()) + " "
            s += "\n"
        s+= "\n"

        return s

    def from_string(cls, s, ctx = mx.cpu(0)):
        tab = s.split("\n")

        assert tab[0] == "//" + cls.__name__ + " V" + str(cls.VERSION), "This file doesn't fit the format (wrong class or wrong version)"

        tab2 = []
        for chain in tab:
            if "/" not in chain and chain != "":
                tab2.append(chain.strip(" "))

        sizes = tab2.pop(0).split(" ")
        output_size = int(sizes[0])
        input_size = int(sizes[1])

        bias = nd.ndarray.array(tab2.pop(0).split(" "), ctx = ctx)

        assert bias.shape[0] == output_size, "Wrong dimension for the bias."

        weights = nd.full((output_size, input_size), 0, ctx = ctx)

        for i in range(output_size):
            weights[i] = nd.ndarray.array(tab2.pop(0).split(" "), ctx = ctx)

        function = tab2.pop(0).split(" ")
        for i in range(len(function)):
            try:
                function[i] = activation.__dict__[function[i]]
            except KeyError:
                try:
                    function[i] = nd.__dict__[function[i]]
                except KeyError:
                    raise ValueError("Function not found : " + function[i])

        if len(function) == 1:
            function = function[0]
        else:
            assert len(function) == output_size, "Invalid number of function"

        bias_fixed = nd.ndarray.array(tab2.pop(0).split(" "), ctx = ctx, dtype = 'uint8')

        weights_fixed = nd.full((output_size, input_size), 0, ctx = ctx, dtype = 'uint8')

        for i in range(output_size):
            weights_fixed[i] = nd.ndarray.array(tab2.pop(0).split(" "), ctx = ctx, dtype = 'uint8')

        layers = cls(output_size, input_size, bias = bias, weights = weights, function = function, ctx = ctx)

        layers.bias_fixed = bias_fixed
        layers.weights_fixed = weights_fixed

        return layers
    from_string = classmethod(from_string)




    def save(self,file):
        with open(file,"w") as f:
            f.write(self.toString())

    def open(cls, file, ctx = mx.cpu(0)):
        s = ""
        with open(file,"r") as f:
            s = f.read()
        return cls.from_string(s, ctx = ctx)
    open = classmethod(open)
