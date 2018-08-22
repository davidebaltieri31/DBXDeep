# DBXDeep
A didactic C++ implementation of a deep learning library using Eigen that I made a few years ago. Running purely on CPU, multithreaded.
Updated to work with the latest release of the Eigen library (for tensor calculus).

## Requirements
Eigen ( http://eigen.tuxfamily.org/ )
OpenCV ( for the convolution kernel visualization layer)

## What this is
It implements the basic blocks of a convolutional deep neural network, various layers, various variants of SGD a couple of cost functions and a few support classes for loading the MNIST dataset.

## What this isn't
- Fast :)
- A general computational graph library. The networks that can be implemented with this are just stack of computational layers.

## Implemented So Far
- **Layers**:
Fully Connected, Convolution, Batch Normalization, Max Pooling, ReLU, Sigmoid, SoftMax
- **Cost**:
Qadratic, Cross Entropy
- **SGDs**:
SGD, AdaGrad, Modified AdaGrad, AdaDelta, RMSProp, Adam

## Still Missing
Dropout. If I ever have the time I'll implement it, I promise.
