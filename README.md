# xor_pytorch

## A classical XOR neural network using [**pytorch**](https://pytorch.org) in **python3** 

``xor.py`` tested an implementation and the training of
a simple neural network using [pytorch](http://pytorch.org).
The implemented neural network evaluates **XOR** for
two noisy inputs, *A* and *B*.  The classical network
consists of one input and one hidden layers.

```
>>> net = XORNet()
>>> net                                                                     
XORNet (                                                                    
  (fc0): Linear (2 -> 2)                                                    
  (fc1): Linear (2 -> 1)                                                    
)                                                                           
```

The sigmoidal activation mediates all the node inter-connections.
The simple classical two-layer network permits an output which is
basically quadratic in linear combinations of *A* and *B*. The
quaduatic properites arise from the symmetry of *XOR* in *A* and *B*.
However, there are two quadratic solutions corresponding to the
ambiguity of *XOR* for *A = B = 0.5* for a simple two-layer
network.

| solution *0* | solution *1* |
|:----:|:----:|
| *XOR = 0* along *A - B = 0* | *XOR = 1* along *A + B = 1* |
| ![soln0](https://github.com/VC-H/xor_pytorch/blob/master/t0.gif?raw=true) | ![soln1](https://github.com/VC-H/xor_pytorch/blob/master/t1.gif?raw=true) |

The gif outputs are plots of the *XOR(A,B)*
evaluated for a mesh grid of *A* and *B* during the training.

## Training

```
>>> xor = XOR() 
>>> xor.view(p) # initialize the network parameters
>>> # training for solution 0
>>> xor.training(nbatch=25,delta=0.2,table=XORData.TABLE0,save='t0')
>>> xor.view(p) # re-initialize the network parameters
>>> # training for solution 1
>>> xor.training(nbatch=25,delta=0.2,table=XORData.TABLE1,save='t1')
```

``xor.training`` plots and produces a png file for each batch procssed.
``p`` is a set of random weights and biases to initial the network elements.
``XORData.TABLE0`` and ``XORData.TABLE1`` are ``XORData.TRUTHTABLE`` with
the additions of *XOR(0.5,0.5)=0* and *XOR(0.5,0.5)=1* to prime the network
to solution *0* and solution *1* respectively. The random noise in
*A* and *B* is uniform in the range ``[-delta,+delta]``.

## Animations

```shell
$ convert -geometry 400x400 -delay 30 t0_*.png t0.gif
$ convert -geometry 400x400 -delay 30 t1_*.png t1.gif
```
produced the gif files with help of ``convert`` of
[imagemagick](https://www.imagemagick.org/)

## Analysis
``xor.gp`` is a ``gnuplot`` script to illustrate the mathematical
solutions of the network.

# Higher order solutions
Higher order networks seek to introduce higher symmetry to the
solution. A third layer would perform linear combinations of outputs 
from the two-layer feeder networks below. Consider

```python
class XORNet3(nn.Modules):
def __init__(self):
    super(XORNet, self).__init__()
    self.fc0 = nn.Linear(2,8)
    self.fc1 = nn.Linear(8,8)
    self.fc2 = nn.Linear(8,1)
def forward(self,x):
    x = F.sigmoid(self.fc0(x))
    x = F.sigmoid(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    return x
```

``XORNet3`` is a simple upgrade of ``XORNet`` of ``xor.py``.
Training can use ``XORDATA.TRUTHTABLE`` un-modified unlike the
case of the two-layer network. The noise in *A* and *B* takes
the full span of ``[-0.5,0.5]``. (*\*note\* ``XORNet3`` is
not included in the repository*)

| ``XORNet3`` training |
|:----:|
| ![soln](https://github.com/VC-H/xor_pytorch/blob/master/t.gif?raw=true) |
