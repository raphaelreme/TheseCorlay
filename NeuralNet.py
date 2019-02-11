# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd

from Layer import Layer


class NeuralNet:
    """Represent a neural network able to correct a code.

    Can be trained and used on data.
    Can also be stored in file or create from a file"""

    VERSION = 1.1
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



    def train(self,epochs):
        for i in range(epochs):
            efficiency = 0
            cumuLoss = 0
            for j in range(self.nbIter):
                z = nd.round(nd.random.uniform(0,1,(self.batchSize,self.code.k),ctx=self.ctx))
                x = nd.dot(z,self.code.G)%2

                noiseBSC = nd.random.uniform(0.01,0.99,(self.batchSize,self.code.n),ctx=self.ctx)
                noiseBSC = nd.floor(noiseBSC/nd.max(noiseBSC,axis=(1,)).reshape((self.batchSize,1)))
                actif = nd.array([[random.uniform(0,1)>0.125]*self.code.n for k in range(self.batchSize)], ctx = self.ctx)
                noiseBSC = noiseBSC * actif


                y = (x + noiseBSC)%2

                with autograd.record():
                    zHat = self.net(y)
                    loss = self.SE(zHat,z)
                loss.backward()

                self.adam(self.params,self.vs,self.sqrs, self.maximums, self.lr, self.batchSize, self.t)
                self.t+=1

                cumuLoss += loss.asscalar()
                zHat = nd.round(zHat)
                efficiency += nd.sum(nd.equal(zHat,z)).asscalar()


            Pc = efficiency/(self.batchSize*self.nbIter*self.code.k)
            Pe = 1 - Pc
            normCumuLoss = cumuLoss/(self.batchSize*self.nbIter*self.code.k)
            print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))

    def train2(self,epochs):
        batchSize = self.code.size
        x = self.code.F
        z = self.code.E

        for i in range(epochs):
            efficiency = 0
            cumuLoss = 0
            for j in range(self.nbIter):
                noiseBSC = nd.random.uniform(0.01,0.99,(batchSize,self.code.n),ctx=self.ctx)
                noiseBSC = nd.floor(noiseBSC/nd.max(noiseBSC,axis=(1,)).reshape((batchSize,1)))
                actif = nd.array([[random.uniform(0,1)>0.125]*self.code.n for k in range(batchSize)], ctx = self.ctx)
                noiseBSC = noiseBSC * actif

                y = (x + noiseBSC)%2

                with autograd.record():
                    zHat = self.net(y)
                    loss = self.SE(zHat,z)
                loss.backward()

                self.adam(self.params,self.vs,self.sqrs, self.maximums, self.lr, batchSize, self.t)
                self.t+=1

                cumuLoss += loss.asscalar()
                zHat = nd.round(zHat)
                efficiency += nd.sum(nd.equal(zHat,z)).asscalar()


            Pc = efficiency/(batchSize*self.nbIter*self.code.k)
            Pe = 1 - Pc
            normCumuLoss = cumuLoss/(batchSize*self.nbIter*self.code.k)
            print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))

    def train3(self,epochs):
        batchSize = len(self.code.reachableWords)
        z = []
        for elt in self.code.E.asnumpy():
            z.extend([list(elt)]*(self.code.n+1))
        z = nd.array(z,ctx=self.ctx)
        x = self.code.reachableWords

        for i in range(epochs):
            efficiency = 0
            cumuLoss = 0
            for j in range(self.nbIter):
                with autograd.record():
                    zHat = self.net(x)
                    loss = self.SE(zHat,z)
                loss.backward()

                self.adam(self.params,self.vs,self.sqrs, self.maximums, self.lr, batchSize, self.t)
                self.t+=1

                cumuLoss += loss.asscalar()
                zHat = nd.round(zHat)
                efficiency += nd.sum(nd.equal(zHat,z)).asscalar()

            Pc = efficiency/(batchSize*self.nbIter*self.code.k)
            Pe = 1 - Pc
            normCumuLoss = cumuLoss/(batchSize*self.nbIter*self.code.k)
            print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))



    def computePerformances(self):
        wrong = []
        cpt = 0
        for i in range(len(self.code.E)):
            for word in self.code.reachableWords[(self.code.n+1)*i:(self.code.n+1)*(i+1)]:
                zhat = self.net(word)
                for diff in nd.round(zhat+self.code.E[i]):
                    if diff.asscalar()%2 != 0:
                        wrong.append((self.code.E[i],word))
                        cpt+=1
                        break

        return (cpt,len(self.code.reachableWords),wrong)


    ##overwrite the file ./file
    def save(self,file):
        with open(file,"w") as f:
            f.write(self.to_string())

    def to_string(self):
        s = "//" + type(self).__name__ + " V" + str(self.VERSION) + "\n\n"
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
