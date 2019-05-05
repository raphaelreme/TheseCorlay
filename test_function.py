from NeuralNet import NeuralNet, compute_rot
from Layer import Layer
from mxnet import nd
from activation import *
import matplotlib
import matplotlib.pyplot as plt
# import numpy as np

def triangle(input):
    x = nd.abs(input[0])
    x_floor = nd.floor(x)
    x = nd.where(nd.modulo(x_floor, 2), 1 - x + x_floor, x - x_floor)
    return input[1] - x > 0


def test_triangle(n, norm_vect):
    sym = []

    sym.append((norm_vect,0))
    for i in range(n):
        sym.append((-norm_vect, 10* 2**(n-i-1)))

    folding_net = NeuralNet.folding_net(sym, optimize = True)
    layers = []
    layers.append(Layer(2,2, weights = compute_rot(norm_vect), bias = nd.zeros(2), function = nd.identity))
    layers.append(Layer(1,2, weights = nd.array([[-10, 1]]), bias = nd.array([0]), function = echelon))
    compute_net = NeuralNet([2,2,1], layers)

    size = 2**(n+12)
    inputs = nd.zeros((2, size))
    inputs[0] = nd.random.uniform(-2**(n+1), 2**(n+1), size)
    inputs[1] =  nd.random.uniform(-2**(n+1), 2**(n+1), size)

    outputs = compute_net.compute(folding_net.compute(inputs))

    x = list(inputs[0].asnumpy())
    y = list(inputs[1].asnumpy())
    results = list(outputs.asnumpy()[0])

#    def triangle(x, y):
#        x = nd.abs(x)
#        x_floor = nd.floor(x)
#        x = nd.where(nd.modulo(x_floor, 2), 1 - x + x_floor, x - x_floor)
#        return y - x > 0

#    true_outputs = list(triangle(inputs[0], inputs[1]).asnumpy())

    colors = ['red','green']

    plt.scatter(x, y, c=results, cmap=matplotlib.colors.ListedColormap(colors), marker = '.')
    plt.show()

#    plt.scatter(x,y, c = true_outputs, cmap=matplotlib.colors.ListedColormap(colors), marker = '.')
#    plt.show()

    return sym, folding_net, compute_net

def test_triangle_horiz(n):
    norm_vect = nd.array([1,0])
    sym = []

    sym.append((norm_vect,0))
    for i in range(n):
        sym.append((-norm_vect, 2**(n-i-1)))

    folding_net = NeuralNet.folding_net(sym, optimize = True)
    layers = []
    layers.append(Layer(2,2, weights = compute_rot(norm_vect), bias = nd.zeros(2), function = nd.identity))
    layers.append(Layer(1,2, weights = nd.array([[-1, 1]]), bias = nd.array([0]), function = echelon))
    compute_net = NeuralNet([2,2,1], layers)

    size = 2**(n+12)
    inputs = nd.zeros((2, size))
    inputs[0] = nd.random.uniform(-2**(n), 2**(n), size)
    inputs[1] =  nd.random.uniform(-0.1, 1.1, size)

    outputs = compute_net.compute(folding_net.compute(inputs))

    def triangle(x, y):
        x = nd.abs(x)
        x_floor = nd.floor(x)
        x = nd.where(nd.modulo(x_floor, 2), 1 - x + x_floor, x - x_floor)
        return y - x > 0
    true_outputs = triangle(inputs[0], inputs[1])

    errors = nd.sum(nd.abs(true_outputs - outputs))

    print("MODEL PROPERTY :")
    print("--------------------------------------")
    print("Number of layers :", folding_net.layersNumber + compute_net.layersNumber)
    print("Number of parameters :", folding_net.size() + compute_net.size())
    print("Errors :", errors, "/", size, "=", errors/size)
    print("--------------------------------------")


if __name__ == "__main__":
    test_triangle(4, nd.array([1,0]))
    #test_triangle_horiz(12)
