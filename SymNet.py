# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd

class SymNet:
    """Represent a very specific NN :
    First learnable symetries. Then a linear classifier.
    Only for a 2D representation yet.

    Can be trained and used on data.
    Can also be stored in file or create from a file

    ctx parameter isn't really implemented yet"""

    VERSION = 1.0
    decoding_class = "SymNet"

    def squared_error(yhat,y):
        return nd.sum((yhat - y) ** 2)
    squared_error = staticmethod(squared_error)


    def __init__(self, nb_of_sym, thetas = None, biases = None, thetas_fixed = None, biases_fixed = None, optimize = True, ctx = mx.cpu(0)): #dim = 2
        assert nb_of_sym >= 1

        self.ctx = ctx
        self.n = nb_of_sym
        self.optimize = optimize

        self.set_thetas(thetas)
        self.set_biases(biases)
        self.set_thetas_fixed(thetas_fixed)
        self.set_biases_fixed(biases_fixed)

        self.init_train()

    def set_biases(self, biases):
        if biases is None:
            self.biases = [nd.random.normal(0,1, ctx = self.ctx) for i in range(self.n)]
        else:
            assert len(biases) == self.n, "Wrong number of biases, should be n."
            self.biases = [bias.copyto(self.ctx) for bias in biases]

    def set_thetas(self, thetas):
        if thetas is None:
            self.thetas = [nd.random.normal(0,1, ctx = self.ctx) for i in range(self.n + 1)] # si dim !=2 on pourrait faire dim - 1 rot donc dim-1*n angle
        else:
            assert len(thetas) == self.n + 1, "Wrong number of thetas, should be n + 1."
            self.thetas = [theta.copyto(self.ctx) for theta in thetas]

    def set_biases_fixed(self, biases_fixed):
        if biases_fixed is None:
            self.biases_fixed = [0] * self.n
        else:
            try:
                if len(biases_fixed) == self.n:
                    self.biases_fixed = [bias for bias in biases_fixed]
                else:
                    raise(ValueError("Wrong number of biases fixed. Should be a list of n elements or a boolean value."))
            except TypeError:
                self.biases_fixed = [bool(biases_fixed)] * self.n

    def set_thetas_fixed(self, thetas_fixed):
        if thetas_fixed is None:
            self.thetas_fixed = [0] * (self.n + 1)
        else:
            try:
                if len(thetas_fixed) == self.n + 1:
                    self.thetas_fixed = [theta for theta in thetas_fixed]
                else:
                    raise(ValueError("Wrong number of thetas fixed. Should be a list of n+1 elements or a boolean value."))
            except TypeError:
                self.thetas_fixed = [bool(thetas_fixed)] * (self.n + 1)



    def size(self):
        pass

    def compute(self,input):
        """Compute the output of the neural net given the input.
        The input has to be a ndarray of shape input_size or (input_size, N) where N is the batch_size."""
        if len(input.shape) == 1:
            input = input.reshape((input.shape[0], 1))

        X = input[0]
        Y = input[1]
        for i in range(self.n):
            _X = nd.cos(self.thetas[i]) * X + nd.sin(self.thetas[i]) * Y # Sym / (cos(O)u_x + sin(O)u_y)
            Y = -nd.sin(self.thetas[i]) * X + nd.cos(self.thetas[i]) * Y

            X = nd.abs(_X + self.biases[i]) - self.biases[i]

            if (False): # optimize
                _X = nd.cos(self.thetas[i]) * X - nd.sin(self.thetas[i]) * Y # Sym / (cos(O)u_x + sin(O)u_y)
                Y = nd.sin(self.thetas[i]) * X + nd.cos(self.thetas[i]) * Y
                X = _X

        _X = nd.cos(self.thetas[self.n]) * X + nd.sin(self.thetas[self.n]) * Y
        Y = -nd.sin(self.thetas[self.n]) * X + nd.cos(self.thetas[self.n]) * Y

        return nd.sigmoid(Y-_X)

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
                    loss = SymNet.squared_error(output, batch[:, inputs.shape[0]:].T)
                loss.backward()

                self.adam_descent(batch_size, lr)

                output = nd.round(output)
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
        for theta in self.thetas:
            theta.attach_grad()
        for bias in self.biases:
            bias.attach_grad()

        self.t = 1

        self.v_biases = [nd.zeros(1, ctx = self.ctx) for i in range(self.n)]
        self.sqr_biases = [nd.zeros(1, ctx = self.ctx) for i in range(self.n)]
        self.v_thetas = [nd.zeros(1, ctx = self.ctx) for i in range(self.n+1)]
        self.sqr_thetas = [nd.zeros(1, ctx = self.ctx) for i in range(self.n+1)]

    def adam_descent(self, batch_size = 32, lr = 0.001):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8

        for i in range(self.n):
            g = self.biases[i].grad/batch_size
            self.v_biases[i] = beta1 * self.v_biases[i] + (1. - beta1) * g
            self.sqr_biases[i] = beta2 * self.sqr_biases[i] + (1. - beta2) * nd.square(g)

            v_bias_corr = self.v_biases[i]/(1. - beta1**self.t)
            sqr_bias_corr = self.sqr_biases[i]/(1. - beta2**self.t)
            div = lr * v_bias_corr/(nd.sqrt(sqr_bias_corr) + eps_stable)

            self.biases[i][:] = self.biases[i] - (1 - self.biases_fixed[i]) * div
            #self.bias = nd.where(self.bias > maximum, maximum, self.bias)
            #self.bias = nd.where(self.bias < -maximum, -maximum, self.bias)

        for i in range(self.n + 1):
            g = self.thetas[i].grad/batch_size
            self.v_thetas[i] = beta1 * self.v_thetas[i] + (1. - beta1) * g
            self.sqr_thetas[i] = beta2 * self.sqr_thetas[i] + (1. - beta2) * nd.square(g)

            v_bias_corr = self.v_thetas[i]/(1. - beta1**self.t)
            sqr_bias_corr = self.sqr_thetas[i]/(1. - beta2**self.t)
            div = lr * v_bias_corr/(nd.sqrt(sqr_bias_corr) + eps_stable)

            self.thetas[i][:] = self.thetas[i] - (1 - self.thetas_fixed[i]) * div

        self.t += 1


    def save(self,file):
        """Save to the file of path ./file.
        Be careful, it overwrites the file."""

        with open(file,"w") as f:
            f.write(self.to_string())

    def to_string(self):
        s = "//" + type(self).decoding_class + " V" + str(type(self).VERSION) + "\n\n"


        s += "//thetas :\n"
        for theta in self.thetas:
            s += str(theta.asscalar()) + " "

        s+= "\n\n//Biases :\n"
        for bias in self.biases:
            s += str(bias.asscalar()) + " "


        s+= "\n\n//Thetas fixed :\n"
        for theta in self.thetas_fixed:
            s += str(theta) + " "

        s+= "\n\n//Biases fixed:\n"
        for bias in self.biases_fixed:
            s += str(bias) + " "

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
        tab = s.split("\n")

        assert tab[0] == "//" + cls.__name__ + " V" + str(cls.VERSION), "This file doesn't fit the format (wrong class or wrong version)"

        tab2 = []
        for chain in tab:
            if "/" not in chain and chain != "":
                tab2.append(chain.strip(" "))

        thetas = []
        for chain in tab2.pop(0).split(" "):
            thetas.append(nd.array([float(chain)]))

        biases = []
        for chain in tab2.pop(0).split(" "):
            biases.append(nd.array([float(chain)]))

        thetas_fixed = []
        for chain in tab2.pop(0).split(" "):
            thetas_fixed.append(int(chain))

        biases_fixed = []
        for chain in tab2.pop(0).split(" "):
            biases_fixed.append(int(chain))

        return cls(len(biases), thetas, biases, thetas_fixed, biases_fixed, ctx = ctx)
    from_string = classmethod(from_string)
