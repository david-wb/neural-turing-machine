# neural-turing-machine-explained
This repo contains a minimalistic TensorFlow implementation of the Neural Turing Machine model, 
the first of the so called "memory augmented neural networks" or "memory networks" for short.
The implementation is intentionally kept as simple as possible with zero fluff so that even a caveman
like me can understand it.

This model was trained on a copy task similar to the one from the original paper. In the copy task,
the NTM is fed a sequence of random bits one at a time, followed by a sequence of -1's of the same length. 
The model is trained to output the original sequence bits when it receives the -1's. 

The main components of the NTM architecture are shown here:

![Alt text](./static/ntm.png?raw=true "NTM Model Diagram")

In this implementation, the controller is a simple feed-forward NN, which takes one external output,
the bits in a sequence, and one internal input, the previous read vector from the memory.

The memory itself is just a matrix with 128 rows and  16 columns. Before the start of each new sequence,
the memory matrix is filled with a small constant value 1e-6. 

Every time the model receives an external bit, it first passes the bit and the previous read vector to
the controller. The controller has two outputs, an external output which is just a single sigmoid value,
and an internal output, a vector of length 100, which is passed to the read and write heads. 

The read and write heads each pass the controller vector through a dense layer to compute the parameters
that they need address the memory and perform reads and writes. For the read head, these are 

```python
k, beta, g, gamma, s
```

where `k` is a keyvector for content-based addressing. `s` is a shift vector of length 3 for moving the up
and down or keeping it in place. `g` determines how much the new weight vector is weighted over the previous one:

```python
w = g * wc + (1 - g) * w_prev
``` 
