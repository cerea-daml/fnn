
# The Fortran Neural Network (FNN) library

The goal of the FNN library is to provide the `fnn` module, which can be used
in Fortran code to implement simple, sequential neural networks. Once a
network is constructed, the **forward** operator is available with
`apply_forward` and can be applied both in training and inference
mode. The **tangent linear** and **adjoint** operators are also available.
In addition, it is possible to access (both read
and write) the networks parameters as a real vector.

This module has been largely inspired by the 
[FKB](https://github.com/scientific-computing/FKB).
It has been designed to provide the technical possibility to use neural
networks within variational data assimilation using the Object-Oriented 
Prediction System (OOPS) developed at the European Centre for Medium-range 
Weather Forecast (ECMWF).

This repository is organised as follows.
- The source code is located in the `src` folder. The `.f90` files together
implement the `fnn` module. The python script is used to create text files
(see Section *Creating a network*).
- The test suite is located in the `test` folder.
- The module documentation is located in the `doc` folder.

## Floating-point precision

The precision for both real and integer numbers is defined in `fnn_common.f90`.
For compatibility with OOPS, these are hard-coded to real64 and int32.
Nevertheless, there is no obstacle to use this library with other kind of real
and integer numbers.

## Creating a network

The easiest way to create a network with the `fnn` module is to use the
`snn_fromfile` function which can read well-formatted text files.
The `keras_to_fnn.py` script can be used to transform a sequential
neural network implemented in Keras into a well-formatted text file.

## Test suite

The test suite consists of a set of python scripts testing various functionalities
of the module. The tests must be compiled (for example using `scons`) before they
can be executed.

## Examples

In the `test/` folder, several examples are provided. Let us take the example
of test 3. The workflow of this test is handled by the python script `test_3.py`.
1. A sequential neural network is created using Keras.
2. The network is converted to a text file using the `keras_to_fnn.py` script.
3. The fortran program `test_3.x` is executed. Note that this program has to be
compiled from `test_3.f90` beforehand, for example using `scons`.
    - The text file is read to construct the network.
    - An ensemble of 100 random input vectors is drawn.
    - The forward operator is applied to each of the 100 random inputs.
    - The fortran input and outputs are written to a binary file.
4. Back to the python script, the binary file is read, and the fortran
output are compared to their expected value.

