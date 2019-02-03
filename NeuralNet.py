# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd

from Layers import Layers


class NeuralNet:
    """Represent a neural network able to correct a code.

    Can be trained and used on data.
    Can also be stored in file or create from a file"""

    ###########Training characteristics and methods####################
    nbIter = 50
    batchSize = 500

    def SE(yhat,y):
        return nd.sum((yhat - y) ** 2)
    SE = staticmethod(SE)

    ##pourrait etre une methode d'objet
    def adam(params, vs, sqrs, maximums, lr, batch_size, t):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8

        for param, v, sqr, maximum in zip(params, vs, sqrs, maximums):
            if param.grad is None:
                continue
            g = param.grad / batch_size

            v[:] = beta1 * v + (1. - beta1) * g
            sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

            v_bias_corr = v / (1. - beta1 ** t)
            sqr_bias_corr = sqr / (1. - beta2 ** t)

            div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
            param[:] = param - div
            param[:] = nd.where(param > maximum, maximum, param)
            param[:] = nd.where(param < -maximum, -maximum, param)



    adam = staticmethod(adam)
    ###############################################




    def __init__(self, layersNumber, sizes = None, layers = None, ctx = None, fct = ampliOP, maximum = 1):
        self.ctx = ctx
        if ctx:
            self.ctx = ctx


        self.fct = fct

        self.layersNumber = insideLayersNumber
        self.sizes = sizes
        if sizes == None:
            #Mettre les

        self.params = list()
        self.maximums = []

        for i in range(self.layersNumber+1):
            self.params.append(nd.random_normal(loc = 0,scale = 0.05,shape=(self.sizes[i],self.sizes[i+1]),ctx=self.ctx))
            self.params.append(nd.random.normal(loc = 0,scale = 0.05, shape = (self.sizes[i+1],),ctx=self.ctx))
            self.maximums.append(nd.array([[maximum for l in range(self.sizes[i+1])] for k in range(self.sizes[i])]))
            self.maximums.append(nd.array([maximum for l in range(self.sizes[i+1])]))


        self.t = 1
        self.vs = []
        self.sqrs = []

        for param in self.params:
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())

        self.lr = 0.001


    def size(self):
        S = 0
        for param in self.params:
            S+= param.size
        return S

    ##Peut etre rajouter une autre methode qui fait le round en sortie
    def net(self,input):
        L = input
        for i in range(self.layersNumber+1):
            L = self.fct(nd.dot(L,self.params[2*i])+self.params[2*i+1])
        return L



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
            f.write(self.toString())

    def toString(self):
        s = "/Dimension :\n\n"
        s += str(self.layersNumber) + "\n"
        for size in self.sizes:
            s += str(size) + " "

        G = self.code.G.asnumpy()
        s += "\n\n/Code :\n"
        for ligne in G:
            s+="\n"
            for value in ligne:
                s+=str(value) + " "


        s+="\n\n/Parameters :\n"

        for param in self.params:
            s+="\n"
            for ligne in param:
                for value in ligne:
                    s+= str(value.asnumpy()[0]) + " "
                s+="\n"
        return s


    def open(cls, file, ctx = mx.cpu(0)):
        s = ""
        with open(file,"r") as f:
            s = f.read()
        return cls.stringToNet(s, ctx = ctx)
    open = classmethod(open)

    def stringToNet(cls, s, ctx = mx.cpu(0)):
        tab = s.split("\n")
        tab2 = []
        for chain in tab:
            if "/" not in chain and chain != "":
                tab2.append(chain.strip(" "))

        insideLayersNumber = int(tab2.pop(0))
        sizes = []

        for chain in tab2.pop(0).split(" "):
            sizes.append(int(chain))
        k = sizes[-1]
        n = sizes[0]
        G = nd.array([[0 for j in range(n)] for i in range(k)],ctx=mx.cpu(0))

        for i in range(k):
            ligne = tab2.pop(0).split(" ")
            for j in range(n):
                G[i,j] = float(ligne[j])

        code = Code(G, ctx)
        net = cls(code,insideLayersNumber,sizes[1:-1], ctx)

        for param in net.params:
            for ligne in param:
                ligne_fichier = tab2.pop(0).split(" ")
                for i in range(len(ligne_fichier)):
                    ligne[i] = float(ligne_fichier[i])

        return net
    stringToNet = classmethod(stringToNet)
