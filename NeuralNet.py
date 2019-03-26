# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd

from Layer import Layer

def compute_rot(v):
    """Return the rotationnal matrix M so that M.v = ||v||e1."""
    if v[0] >= 0:
        M = nd.eye(len(v))
    else:
        M = - nd.eye(len(v))
    for i in range(1, len(v)):
        if v[i] == 0:
            continue
        rot_minus_theta = nd.eye(len(v))
        temp = nd.dot(M, v)

        theta = nd.arctan(temp[i]/temp[0])
        c = nd.cos(theta)
        s = nd.sin(theta)

        rot_minus_theta[0,0] = c
        rot_minus_theta[i,i] = c
        rot_minus_theta[0,i] = s
        rot_minus_theta[i,0] = -s

        M = nd.dot(rot_minus_theta, M)
    return M

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


    def __init__(self, sizes, layers = None, maximum = 1, ctx = mx.cpu(0)):
        assert len(sizes) >= 2 #At least two sizes : the input and output sizes

        self.sizes = sizes
        self.layersNumber = len(sizes) - 1

#        try:
#            assert len(fct) == self.layersNumber, "Function argument has to be one fonction (for all layers) or a function (or a valid set) for each layers"
#        except TypeError:
#            fct = [fct]*self.layersNumber

        self.layers = [None] * self.layersNumber
        if layers is not None:
            for i in range(self.layersNumber):
                self.set_layer(layers[i], i)
        else:
            for i in range(self.layersNumber):
                self.set_layer(Layer(sizes[i+1], sizes[i], ctx = ctx), i)

    def set_layer(self, layer, i):
        assert i < self.layersNumber, "Wrong index. Must be between 0 and layersNumber-1."
        assert layer.input_size == self.sizes[i] and layer.output_size == self.sizes[i+1], "The sizes of the layer don't fit the NeuralNet one's."
        self.layers[i] = layer

    def size(self):
        S = 0
        for layer in self.layers:
            S+= layer.total_size
        return S


    def compute(self,input):
        """Compute the output of the neural net given the input.
        The input has to be a ndarray of shape input_size or (input_size, N) where N is the batch_size."""

        for layer in self.layers:
            input = layer.compute(input)
        return input

    def train(self, inputs, outputs, epochs = 10, batch_size = 32, lr = 0.001, transform = None, verbose = True):
        """train the neural network to fit the outputs with the inputs.

        Args:
            inputs: an ndarray of input.
            outputs: an ndarray of outputs.
            epochs, batch_size, lr: the parameters of the learning algorithm.
            transform: if None, take the output as given, else try to compute
                        transformed outputs = transform(outputs) and fit with them.
            verbose: If True then the results will be displayed all along the training.
        Returns:
            The historical of the training. (tuple of array)."""

        if transform:
            outputs = transform(outputs)
        n = (inputs.shape[1]-1)//batch_size + 1

        #inputs-1/batch - 1 < n <= inputs-1/batch
        if len(outputs.shape) == 1:
            outputs = outputs.reshape((1, outputs.shape[0]))
        assert inputs.shape[1] == outputs.shape[1], "Shapes does not match."

        data = nd.concat(inputs.T, outputs.T)

        efficiencies = []
        cumuLosses = []
        epochs = list(range(epochs))

        for i in epochs:
            efficiency = 0
            cumuLoss = 0
            data = nd.shuffle(data)
            batchs = [data[k*batch_size:min(inputs.shape[1], (k+1)*batch_size),:] for k in range(n)]
            for batch in batchs:
                with autograd.record():
                    output = self.compute(batch[:, :inputs.shape[0]].T)
                    loss = NeuralNet.squared_error(output, batch[:, inputs.shape[0]:].T)
                loss.backward()

                self.adam_descent(batch_size, lr)

                cumuLoss += loss.asscalar()
                efficiency += nd.sum(nd.equal(output, batch[:, inputs.shape[0]:].T)).asscalar()


            efficiency /= outputs.shape[1] * outputs.shape[0]
            efficiencies.append(efficiency)

            cumuLoss /= outputs.shape[1] * outputs.shape[0]
            cumuLosses.append(cumuLoss)

            if verbose:
                print("Epochs %d: Pe = %lf , loss = %lf" % (i,1-efficiency,cumuLoss))

        return (epochs, cumuLosses, efficiencies)

    def init_train(self):
        for layer in self.layers:
            layer.init_grad()

    def adam_descent(self, batch_size = 32, lr = 0.001):
        for layer in self.layers:
            layer.adam_descent(batch_size, lr)


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

    ####################################################
    ######## CLASS METHODS -> Create NeuralNet #########
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
            net.set_layer(Layer.from_string(tab[i], ctx), i)

        return net
    from_string = classmethod(from_string)



    def folding_net(cls, symetries, ctx = mx.cpu(0), optimize = False):
        """NeuralNet generated by a set of symetries.
        Given the symetries, it computes the net which folds the space according to them.

        The symetries considered are ONLY the orthogonal reflections accros hyperplanes.
        Let the equation of the hyperplane be n.x + b = 0. Then the representation of the symetry
        is the tupple (n, b).

        One should notice than (-n, -b) represent the same hyperplane. But not the same operation for the NeuralNet :
        It will fold the space into the subspace pointed by n (where x.n + b >= 0)

        A list of tupple (n, b) is expected as symetries
        The optimize parameter allows you to break up the operations in more layers.
        """
        layers = []

        #For optimization
        n = symetries[0][0]
        weights_rest = nd.eye(len(n))
        bias_rest = nd.zeros(len(n))

        for n, b in symetries:
            ROT = compute_rot(n)
            if optimize:
                weights_rest = nd.dot(ROT, weights_rest)
                bias_rest = nd.dot(ROT, bias_rest)
            else:
                layers.append(Layer(len(n), len(n), weights = ROT, bias = nd.zeros(len(n)), function = nd.identity, fixed = True, ctx = ctx))

            # Symetry for the plane x = -b
            # n1' = |n1 + b| - b. ni' = ni.
            function = [nd.relu]*2 + [nd.identity] * (len(n)-1)
            weights = nd.eye(len(n)+1, len(n), -1)
            weights[0,0] = -1
            bias = nd.array([-b,b] + [0] * (len(n) - 1)) + nd.dot(weights, bias_rest)
            weights = nd.dot(weights, weights_rest)

            layers.append(Layer(len(n)+1, weights.shape[1] , weights = weights, bias = bias, function = function, fixed = True, ctx = ctx))

            bias = nd.array([-b] + [0] * (len(n)-1))
            weights = nd.eye(len(n), len(n)+1, 1)
            weights[0,0] = 1

            if optimize:
                weights_rest = nd.dot(ROT.T, weights)
                bias_rest = nd.dot(ROT.T, bias)
            else:
                layers.append(Layer(len(n), len(n)+1, weights = weights, bias = bias, function = nd.identity, fixed = True, ctx = ctx))
                layers.append(Layer(len(n), len(n), weights = ROT.T, bias = nd.zeros(len(n)), function = nd.identity, fixed = True, ctx = ctx))

        if optimize:
            layers.append(Layer(len(n), len(n)+1, weights = weights_rest, bias = bias_rest, function = nd.identity, fixed = True, ctx = ctx))

        sizes = [layers[0].input_size]
        for layer in layers:
            sizes.append(layer.output_size)

        return cls(sizes, layers, ctx = ctx)
    folding_net = classmethod(folding_net)
