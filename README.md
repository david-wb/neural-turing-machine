# neural-turing-machine-explained
This repo contains a minimalistic implementation of the Neural Turing Machine model, 
the first of the so called "memory augmented neural networks" or "memory networks" for short.
The implementation is intentionally kept as simple as possible with zero fluff so that even a caveman
like me can understand it.

This model was trained on two tasks from the original paper: the copy task, and an associative recall task. In the copy task,
the NTM is fed a sequence of random bits one at a time, followed by a sequence of -1's of the same length. 
The model is trained to output the original sequence bits when it receives the -1's. 