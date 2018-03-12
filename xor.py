#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import FloatTensor
from torch.autograd import Variable


fig = mp.figure(figsize=(4,4))
ax3d = mp.axes(projection='3d')
mp.ion()


class XORData(object):
    """a class for the generation of XOR validation and training data

    >>> d = XORData()
    >>> d.astype(int)
    array([[0, 0, 0],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]])

    >>> d = XORData(batchsize=2,delta=0.5)
    >>> len(d)
    8
    >>> np.all(np.rint(d[0:4]) == XORData.TRUTHTABLE)
    True
    >>> np.all(np.rint(d[4:8]) == XORData.TRUTHTABLE)
    True
    >>> np.var(d - np.vstack([XORData.TRUTHTABLE]*2)) > 0
    True

    """

    TRUTHTABLE = np.array([
        #A,B,XOR
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,0],
        ],dtype=float)

    TABLE0 = np.vstack([TRUTHTABLE,[0.5,0.5,0.0]])
    TABLE1 = np.vstack([TRUTHTABLE,[0.5,0.5,1.0]])

    def __new__(this,batchsize=1,delta=0.0,table=TRUTHTABLE):
        n = len(table)
        assert table.shape == (n,2+1)
        rands = np.random.uniform(-delta,+delta,size=(batchsize,n,2))
        zeros = np.zeros(shape=(batchsize,n,1),dtype=float)
        deltas = np.concatenate((rands,zeros),axis=2)
        assert deltas.shape == (batchsize,n,3)
        dataset = table + deltas
        dataset.shape = (batchsize*n,3)
        return dataset


class XORNet(nn.Module):
    """A classical 2-layer XOR neural network

    >>> net = XORNet()
    >>> net
    XORNet (
      (fc0): Linear (2 -> 2)
      (fc1): Linear (2 -> 1)
    )

    """

    def __init__(self):
        super(XORNet, self).__init__()
        self.fc0 = nn.Linear(2,2)
        self.fc1 = nn.Linear(2,1)

    def forward(self,x):
        x = F.sigmoid(self.fc0(x))
        return F.sigmoid(self.fc1(x))

    def setparams_zeros(self):
        for p in self.parameters():
            p.data.zero_()

    def setparams_uniforms(self,delta=1):
        for p in self.parameters():
            p.data.uniform_(-delta,+delta)


class XOR(object):
    """An encapsulation of a neural network, training and testing

    >>> xor = XOR()
    >>> xor
    XOR (
      loss: MSELoss
      optim: Adam
      lr: 0.01
    )

    """

    LEARNING_RATE = 0.01

    def __init__(self,lr=LEARNING_RATE):
        self.net = XORNet()
        self.state_start = self.net.state_dict()
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.net.parameters(),lr)
        self.l = self.training # shorthand

    def training(
            self,nbatch=10,batchsize=100,
            delta=0.2,table=XORData.TRUTHTABLE,
            save=False):
        for ibatch in range(nbatch):
            epsilonsum = 0
            for t in XORData(batchsize,delta,table):
                y = self.net(Variable(FloatTensor(t[0:2])))
                target = Variable(FloatTensor(t[2:]))
                self.optim.zero_grad()
                epsilon = self.loss(y,target)
                epsilonsum += epsilon.data[0]
                epsilon.backward()
                self.optim.step()
            self.splot()
            if save:
                fmt = save + '{:0' + str(len(str(nbatch))) + '}'
                mp.savefig(fmt.format(ibatch))
            print('{:<8} {:.4e}'.format(ibatch,epsilonsum/batchsize))

    def test(self):
        """print the truth table evaluated by self.net:"""
        for a,b,xor in XORData():
            y = self.net(Variable(FloatTensor([a,b])))
            target = Variable(FloatTensor([xor]))
            epsilon = self.loss(y,target)
            print('{} {:+.8f} {:+.8f}'.format(
                (int(a),int(b)),y.data[0],epsilon.data[0]))

    def splot(self,nticks=51):
        """surface plot of the xor outputs of
        the self.net for a mesh grid inputs of a and b:"""
        i = np.linspace(-0.5,1.5,nticks)
        a,b = np.meshgrid(i,i)
        ab = np.stack([a,b],axis=-1)
        xor = self.net(Variable(FloatTensor(ab)))
        xor = xor.data.numpy()
        xor.shape = (nticks,nticks)
        ax3d.clear()
        ax3d.plot_surface(a,b,xor,cmap='viridis',edgecolor='none')
        ax3d.view_init(elev=30,azim=-60)
        ax3d.set_xticks([0,1]),ax3d.set_xlabel('A')
        ax3d.set_yticks([0,1]),ax3d.set_ylabel('B')
        ax3d.set_zticks([0,1]),ax3d.set_zlabel('XOR')
        mp.draw()
        mp.pause(0.05)

    def __repr__(self):
        return "\n".join([
            'XOR (',
            '  loss: {}'.format(self.loss.__class__.__name__),
            '  optim: {}'.format(self.optim.__class__.__name__),
            '  lr: {}'.format(self.optim.param_groups[0].get('lr')),
            ')',
            ])


if __name__ == "__main__":

    import sys
    import doctest

    def docscript(obj=None):
        """usage: exec(docscript())"""
        doc = __doc__
        if obj is not None:
            doc = obj.__doc__
        return doctest.script_from_examples(doc)

    if sys.argv[0] == "": # if python is in an emacs buffer:
        print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))

    state0_dict = OrderedDict((
        ('fc0.weight',FloatTensor([[20,-20],[20,-20]])),
        ('fc0.bias',FloatTensor([-15,15])),
        ('fc1.weight',FloatTensor([[20,-20]])),
        ('fc1.bias',FloatTensor([10])),
        ))

    state1_dict = OrderedDict((
        ('fc0.weight',FloatTensor([[20,20],[20,20]])),
        ('fc0.bias',FloatTensor([-35,-5])),
        ('fc1.weight',FloatTensor([[20,-20]])),
        ('fc1.bias',FloatTensor([10])),
        ))

    state_dict = OrderedDict((
        ('fc0.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc0.bias',FloatTensor([-0.3, 0.5])),
        ('fc1.weight',FloatTensor([[0.4, 0.0]])),
        ('fc1.bias',FloatTensor([-0.4])),
        ))

    # some shorthands
    t = XORData.TRUTHTABLE
    t0 = XORData.TABLE0
    t1 = XORData.TABLE1
