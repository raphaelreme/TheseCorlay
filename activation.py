# -*- coding: utf-8 -*-
from mxnet import nd

def ampliOP(x):
    #return 0.01*(nd.relu(0.1*(x+10))-nd.relu(0.1*(x+0.5))) + 0.98*(nd.relu(x+0.5)-nd.relu(x-0.5)) + 0.01*(nd.relu(0.1*(x-1))-nd.relu(0.1*(x-11)))
    return nd.relu(x+0.5)-nd.relu(x-0.5)

def ampliOPSmooth(x):
    b1 = x<-0.4
    b2 = x>0.4
    return b1*0.1*(nd.exp(x+0.4)) + (1-b1)*(1-b2)*(x+0.5) + b2*(0.9+0.1*(1-nd.exp(-(x-0.4))))

def sigmoid(x):
    return nd.sigmoid(4*x)

def echelon(x):
    return x>0

def echelonSmooth(x):
    return nd.sigmoid(10*x)


#import matplotlib.pyplot as plt
#x = [float(k)/1000 for k in range(-3000,3000)]
#ndx = [nd.array([x[k]]) for k in range(len(x))]
#y = [ampliOPSmooth(ndx[k]).asscalar() for k in range(len(x))]

#plt.plot(x,y)
#plt.show()
