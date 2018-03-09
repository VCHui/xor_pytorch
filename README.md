# xor_pytorch

## A classical XOR neural network using [**pytorch**](https://pytorch.org) in **python3** 

``xor_ö.py`` tested an implementation and the training of
a simple neural network using [pytorch](http://pytorch.org).
The implemented neural network evaluates the **XOR** for two noisy inputs.
The classical network consists of one input and one hidden layers.

```python
>>> net = XORNet()
>>> net                                                                     
XORNet (                                                                    
  (fc0): Linear (2 -> 2)                                                    
  (fc1): Linear (2 -> 1)                                                    
)                                                                           
```

The sigmoidal activation mediates all the node inter-connections.
Let *A* and *B* be the inputs. The classical two-layer network permits
an output which is basically quadratic in *A* and *B*. There are two
solutions of the quadratic corresponding to the ambiguity of
*XOR* for *A = B = 0.5*. The gif outputs are plots of the *XOR(A,B)*
evaluated for a mesh grid of *A* and *B* during the training.

| solution *0* | solution *1* |
|:----:|:----:|
| *XOR = 0* along *A - B = 0* | *XOR = 1* along *A + B = 1* |
| ![soln0](https://github.com/VC-H/xor_pytorch/blob/master/t0.gif?raw=true)| ![soln1](https://github.com/VC-H/xor_pytorch/blob/master/t1.gif?raw=true) |

## Training

```python
>>> xor = XOR()
>>> xor.view(p)
>>> xor.training(nbatch=25,table=XORData.TABLE0,save='t0_')
>>> xor.view(p)
>>> xor.training(nbatch=25,table=XORData.TABLE1,save='t1_')
```
produced a png file for each plot. ``p`` is a set of random weights and
biases to initial the network elements. ``XORData.TABLE0`` and
``XORData.TABLE1`` are ``XORData.TRUTHTABLE`` with the additions of
*XOR(0.5,0.5)=0* and *XOR(0.5,0.5)=1* to train the network to
solution *0* and solution *1* respectively.

```shell
$ convert -geometry 400x400 -delay 30 t0_*.png t0.gif
$ convert -geometry 400x400 -delay 30 t1_*.png t1.gif
```
produced the gif files

## Analysis
``xor.gp`` is a ``gnuplot`` script to illustrate the mathematical
analysis for the solutions of the network.
