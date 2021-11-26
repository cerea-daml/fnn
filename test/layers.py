#!/usr/bin/env python

import numpy as np
from activation import construct_activation

def fromofile(f):
    layer_name = f.readline().strip()
    if layer_name == 'dense':
        Nin = int(f.readline().strip())
        Nout = int(f.readline().strip())
        p = np.loadtxt(f, max_rows=1)
        activation = f.readline().strip()
        layer = DenseLayer(Nin, Nout, activation, initialisation='value', initialisation_kwargs=dict(value=p))
        return layer
    elif layer_name == 'normalisation':
        Ninout = int(f.readline().strip())
        alpha = float(f.readline().strip())
        beta = float(f.readline().strip())
        layer = NormalisationLayer(Ninout, alpha, beta)
        return layer
    else:
        print('unknown layer type:', layer_name)

def construct_layer(name, *args, **kwargs):
    layer_class = dict(
            dense=DenseLayer,
            normalisation=NormalisationLayer,
            )
    return layer_class[name](*args, **kwargs)

class NormalisationLayer:

    def __init__(self, Ninout, alpha, beta):
        self.Nin = Ninout
        self.Nout = Ninout
        self.alpha = alpha
        self.beta = beta
        self.num_parameters = 0
        self.parameters = np.zeros(0)

    def initialise(self, *args, **kwargs):
        pass

    def apply(self, x):
        return self.alpha * x + self.beta

    def apply_linearise(self, x):
        return self.apply(x)

    def apply_tangent_linear_x(self, dx):
        return self.alpha * dx

    def apply_adjoint_x(self, dy):
        return self.alpha * dy

    def apply_tangent_linear_p(self, dp):
        return np.zeros(self.Nout)

    def apply_adjoint_p(self, dy):
        return np.zeros(0)

class DenseLayer:

    def __init__(self, Nin, Nout, activation='linear', activation_kwargs=None, initialisation='randn', initialisation_kwargs=None):
        self.Nin = Nin
        self.Nout = Nout
        self.num_parameters = self.Nout * (self.Nin+1)
        self.parameters = np.empty(self.num_parameters)
        self.b = self.parameters[:self.Nout]
        self.w = self.parameters[self.Nout:].reshape((self.Nout, self.Nin), order='F')

        initialisation_kwargs = initialisation_kwargs or {}
        self.initialise(initialisation, **initialisation_kwargs)

        activation_kwargs = activation_kwargs or {}
        self.activation = construct_activation(activation, **activation_kwargs)

    def initialise(self, initialisation, **kwargs):
        if initialisation == 'zero':
            self.parameters[:] = 0
        elif initialisation == 'randn':
            self.parameters[:] = np.random.randn(self.num_parameters)
        elif initialisation == 'value':
            self.parameters[:] = kwargs['value']

    def apply(self, x):
        z = self.w@x + self.b
        return self.activation.apply(z)

    def apply_linearise(self, x):
        self.x = x.copy()
        z = self.w@x + self.b
        return self.activation.apply_linearise(z)

    def apply_tangent_linear_x(self, dx):
        return self.activation.apply_tangent_linear(self.w@dx)

    def apply_adjoint_x(self, dy):
        return self.w.T@self.activation.apply_adjoint(dy)

    def apply_tangent_linear_p(self, dp):
        db = dp[:self.Nout]
        dw = dp[self.Nout:].reshape((self.Nout, self.Nin), order='F')
        tlw = self.activation.apply_tangent_linear(dw@self.x)
        tlb = self.activation.apply_tangent_linear(db)
        return tlw + tlb

    def apply_adjoint_p(self, dy):
        db = self.activation.apply_adjoint(dy)
        dw = db.reshape((self.Nout, 1))@self.x.reshape((1, self.Nin))
        return np.concatenate([db, dw.flatten(order='F')])

    def gradient_test_x(self, Ntest, eps):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.Nin)
            y = self.apply_linearise(x)
            dx = np.random.randn(self.Nin)
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

    def adjoint_test_x(self, Ntest):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.Nin)
            y = self.apply_linearise(x)
            dx = np.random.randn(self.Nin)
            dy = np.random.randn(self.Nout)
            delta = self.apply_adjoint_x(dy)@dx - self.apply_tangent_linear_x(dx)@dy
            err[t] = abs(delta)
        print('-'*100)
        print('adjoint test x')
        print('Ntest', Ntest)
        print('mean error', err.mean())
        print('max error', err.max())
        print('min error', err.min())

    def gradient_test_p(self, Ntest, eps):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.Nin)
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

    def adjoint_test_p(self, Ntest):
        err = np.zeros(Ntest)
        for t in range(Ntest):
            self.initialise('randn')
            x = np.random.randn(self.Nin)
            y = self.apply_linearise(x)
            dp = np.random.randn(self.num_parameters)
            dy = np.random.randn(self.Nout)
            delta = self.apply_adjoint_p(dy)@dp - self.apply_tangent_linear_p(dp)@dy
            err[t] = abs(delta)
        print('-'*100)
        print('adjoint test p')
        print('Ntest', Ntest)
        print('mean error', err.mean())
        print('max error', err.max())
        print('min error', err.min())

