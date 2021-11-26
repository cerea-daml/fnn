#!/usr/bin/env python

import numpy as np
from layers import construct_layer, fromofile as layer_fromofile

class SequentialNetwork:

    def __init__(self):
        self.layers = []

    def add_layer(self, name, *args, **kwargs):
        self.layers.append(construct_layer(name, *args, **kwargs))

    @property
    def num_parameters(self):
        return sum((layer.num_parameters for layer in self.layers))

    @property
    def parameters(self):
        return self.join_parameters([layer.parameters for layer in self.layers])

    @parameters.setter
    def parameters(self, p):
        p_list = self.split_parameters(p)
        for (layer, pi) in zip(self.layers, p_list):
            layer.parameters[:] = pi

    def split_parameters(self, p):
        p_list = []
        index = 0
        for layer in self.layers:
            p_list.append(p[index:index+layer.num_parameters])
            index += layer.num_parameters
        return p_list

    def join_parameters(self, p_list):
        return np.concatenate(p_list)

    def initialise(self, initialisation):
        for layer in self.layers:
            layer.initialise(initialisation)

    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x

    def apply_linearise(self, x):
        for layer in self.layers:
            x = layer.apply_linearise(x)
        return x

    def apply_tangent_linear_x(self, dx):
        for layer in self.layers:
            dx = layer.apply_tangent_linear_x(dx)
        return dx

    def apply_adjoint_x(self, dy):
        for layer in self.layers[::-1]:
            dy = layer.apply_adjoint_x(dy)
        return dy

    def apply_tangent_linear_p(self, dp):
        dp_list = self.split_parameters(dp)
        dy = self.layers[0].apply_tangent_linear_p(dp_list[0])
        for (layer, dpi) in zip(self.layers[1:], dp_list[1:]):
            dy = layer.apply_tangent_linear_x(dy) + layer.apply_tangent_linear_p(dpi)
        return dy

    def apply_adjoint_p(self, dy):
        dp_list = []
        for layer in self.layers[::-1][:-1]:
            dp_list.append(layer.apply_adjoint_p(dy))
            dy = layer.apply_adjoint_x(dy)
        dp_list.append(self.layers[0].apply_adjoint_p(dy))
        return self.join_parameters(dp_list[::-1])

    def apply_tangent_linear(self, dp, dx):
        dyp = self.apply_tangent_linear_p(dp)
        dyx = self.apply_tangent_linear_x(dx)
        return dyp + dyx

    def apply_adjoint(self, dy):
        dp = self.apply_adjoint_p(dy)
        dx = self.apply_adjoint_x(dy)
        return (dp, dx)

    def gradient_test_x(self, Ntest, eps):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.layers[0].Nin)
            y = self.apply_linearise(x)
            dx = np.random.randn(self.layers[0].Nin)
            dy = self.apply_tangent_linear_x(dx)
            yp = self.apply(x+eps*dx)
            delta = yp-y-eps*dy
            err[t] = abs(delta).max()
        print('-'*100)
        print('gradient test x')
        print('Ntest', Ntest)
        print('mean error', err.mean())
        print('max error', err.max())
        print('min error', err.min())

    def gradient_test_p(self, Ntest, eps):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.layers[0].Nin)
            y = self.apply_linearise(x)
            dp = np.random.randn(self.num_parameters)
            dy = self.apply_tangent_linear_p(dp)
            self.parameters += eps*dp
            yp = self.apply(x)
            delta = yp-y-eps*dy
            err[t] = abs(delta).max()
        print('-'*100)
        print('gradient test p')
        print('Ntest', Ntest)
        print('mean error', err.mean())
        print('max error', err.max())
        print('min error', err.min())

    def adjoint_test_x(self, Ntest):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.layers[0].Nin)
            y = self.apply_linearise(x)
            dx = np.random.randn(self.layers[0].Nin)
            dy = np.random.randn(self.layers[-1].Nout)
            delta = self.apply_adjoint_x(dy)@dx - self.apply_tangent_linear_x(dx)@dy
            err[t] = abs(delta)
        print('-'*100)
        print('adjoint test x')
        print('Ntest', Ntest)
        print('mean error', err.mean())
        print('max error', err.max())
        print('min error', err.min())

    def adjoint_test_p(self, Ntest):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.layers[0].Nin)
            y = self.apply_linearise(x)
            dp = np.random.randn(self.num_parameters)
            dy = np.random.randn(self.layers[-1].Nout)
            delta = self.apply_adjoint_p(dy)@dp - self.apply_tangent_linear_p(dp)@dy
            err[t] = abs(delta)
        print('-'*100)
        print('adjoint test p')
        print('Ntest', Ntest)
        print('mean error', err.mean())
        print('max error', err.max())
        print('min error', err.min())

def fromfile(filename):
    with open(filename, 'r') as f:
        net_name = f.readline().strip()
        if net_name == 'sequential':
            network = SequentialNetwork()
            num_layers = int(f.readline().strip())
            for i in range(num_layers):
                layer = layer_fromofile(f)
                network.layers.append(layer)
            return network
        else:
            print('unknown network type:', net_name)

