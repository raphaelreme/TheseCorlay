# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd

from Layer import Layer


class NeuralNet:
    """Represent a neural network able to correct a code.

    Can be trained and used on data.
    Can also be stored in file or create from a file"""

    VERSION = 1.2
    decoding_class = "NeuralNet"
    #nbIter = 50
    batch_size = 500

    def squared_error(yhat,y):
        return nd.sum((yhat - y) ** 2)
    squared_error = staticmethod(squared_error)


    def __init__(self, sizes, fct = nd.sigmoid, maximum = 1, ctx = mx.cpu(0)):
        assert len(sizes) >= 2 #At least two sizes : the input and output sizes

        self.sizes = sizes
        self.layersNumber = len(sizes) - 1

        try:
            assert len(fct) == self.layersNumber, "Function argument has to be one fonction (for all layers) or a function (or a valid set) for each layers"
        except TypeError:
            fct = [fct]*self.layersNumber


        self.layers = []
        for i in range(self.layersNumber):
            self.layers.append(Layer(sizes[i+1], sizes[i], function = fct[i], fixed = False, ctx = ctx))


    def size(self):
        S = 0
        for layer in self.layers:
            S+= layer.size
        return S


    def compute(self,input):
        """Compute the output of the neural net given the input.
        The input has to be a ndarray of shape input_size or (input_size, N) where N is the batch_size."""

        for layer in self.layers:
            input = layer.compute(input)
        return input


    ##overwrite the file ./file
    def save(self,file):
        """Save to the file of path ./file.
        Be careful, it overwrites the file."""

        with open(file,"w") as f:
            f.write(self.to_string())

    def to_string(self):
        s = "//" + type(self).decoding_class + " V" + str(type(self).VERSION) + "\n\n"
        s += "//Sizes :\n"

        for size in self.sizes:
            s += str(size) +" "
        s+= "\n\n"

        for layer in self.layers:
            s+= "/LAYER\n"
            s+= layer.to_string()

        return s


    def open(cls, file, ctx = mx.cpu(0)):
        s = ""
        with open(file,"r") as f:
            s = f.read()
        return cls.from_string(s, ctx = ctx)
    open = classmethod(open)

    def from_string(cls, s, ctx = mx.cpu(0)):
        tab = s.split("/LAYER\n")
        info_net = tab.pop(0)

        tab_net = info_net.split("\n")

        assert tab_net[0] == "//" + cls.__name__ + " V" + str(cls.VERSION), "This file doesn't fit the format (wrong class or wrong version)"

        tab_net2 = []
        for chain in tab_net:
            if "/" not in chain and chain != "":
                tab_net2.append(chain.strip(" "))

        sizes = []
        for chain in tab_net2.pop(0).split(" "):
            sizes.append(int(chain))

        net = cls(sizes, ctx = ctx)

        assert net.layersNumber == len(tab), "Number of layers doesn't fit the number of sizes in the file."

        for i in range(net.layersNumber):
            net.layers[i] = Layer.from_string(tab[i], ctx)

        return net
    from_string = classmethod(from_string)
