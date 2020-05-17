# The Neural Turing Machine - Explained
This repo contains a bare-minimum implementation of the Neural Turing Machine model, 
the first of the so called "memory augmented neural networks."
It is implemented in TensorFlow and intentionally kept as simple as possible for easy understanding.

The main components of the NTM are shown here:

![Alt text](./static/ntm.png?raw=true "NTM Model Diagram")

The controller is a simple feed-forward NN, which takes one external input,
the bits in a sequence, and one internal input, the previous read vector from the memory.

The memory itself is just a matrix of 100 rows and 20 columns. Before the start of each new sequence,
the memory matrix is filled with a small constant value of 1e-6. 

Every time the model receives an external input, it first passes it and the previous read vector through
a feedword layer. The output of this step is passed to the read-write heads. Finally, the output of the 
first layer is combind with the new read vector and pass through a last fully connected layer to produce the external output.


The read and write heads each pass their input vector through a dense layer to compute the parameters 
they need to address the memory and perform reads and writes. For the read head, these are 

```python
k, beta, g, gamma, s
```

where `k` is a *key* vector for content-based addressing. The key vector is compared with every row of the memory matrix to get a weight vector over rows.
 `s` is a shift vector of length 3 for moving the head up
and down or keeping it in place. `beta` and `gamma` are scalars used essentially for sharpening the weight vector. `g` is the weight given to new weight vector over the previous one:

```python
w = g * w_new + (1 - g) * w_prev
``` 

The write head computes the same parameters and also two vectors `erase` and `add`. These are used to
update each row i of the memory matrix like so:
```python
M[i, :] = M[i, :] * (1 - erase * w[i]) + add * w[i]
```
`erase` and `add` each have the same number of elements as the number of columns in the memory.

## Copy Task

The model was trained on a copy task similar to the one from the original paper. In the copy task,
the NTM is fed a sequence of random bits one at a time, followed by a sequence of -1's of the same length. 
The model is trained to output the original sequence bits when it receives the -1's. Please see the copy task
notebook for more details.

## Training and Evaluation
Train the copy task model using
```python
python train_copy_model.py
```

To test the model out see the `copy_task.ipynb` notebook.
