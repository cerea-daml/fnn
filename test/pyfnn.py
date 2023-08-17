#!/usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np

#--------------------------------------------------
# activation functions
#--------------------------------------------------

def construct_activation(name):
    activation_class = dict(
            linear=LinearActivation,
            tanh=TanhActivation,
            relu=ReluActivation,
            )
    return activation_class[name]()

class AbstractActivation(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def apply(self, z):
        pass

    @abstractmethod
    def apply_linearise(self, z):
        pass

    def apply_tangent_linear(self, dz):
        return self.activation_prime * dz

    def apply_adjoint(self, dy):
        return self.activation_prime * dy

class LinearActivation:

    def __init__(self):
        pass

    def apply(self, z):
        return z

    def apply_linearise(self, z):
        return z

    def apply_tangent_linear(self, dz):
        return dz

    def apply_adjoint(self, dy):
        return dy

class TanhActivation(AbstractActivation):

    def __init__(self):
        super(TanhActivation, self).__init__()

    def apply(self, z):
        return np.tanh(z)

    def apply_linearise(self, z):
        y = np.tanh(z)
        self.activation_prime = 1 - y**2
        return y

class ReluActivation(AbstractActivation):

    def __init__(self):
        super(ReluActivation, self).__init__()

    def apply(self, z):
        return np.maximum(z, 0)

    def apply_linearise(self, z):
        self.activation_prime = 1 * (z>0)
        return np.maximum(z, 0)

#--------------------------------------------------
# layers
#--------------------------------------------------

def layer_fromfile(f_txt, f_bin):
    layer_name = f_txt.readline().strip()
    if layer_name == 'dense':
        Nin = int(f_txt.readline().strip())
        Nout = int(f_txt.readline().strip())
        activation = f_txt.readline().strip()
        p = np.fromfile(f_bin, 'f', count=(Nin+1)*Nout)
        return DenseLayer(Nin, Nout, activation, p)
    elif layer_name == 'frozen-dense':
        Nin = int(f_txt.readline().strip())
        Nout = int(f_txt.readline().strip())
        activation = f_txt.readline().strip()
        p = np.fromfile(f_bin, 'f', count=(Nin+1)*Nout)
        return FrozenDenseLayer(Nin, Nout, activation, p)
    elif layer_name == 'normalisation':
        Ninout = int(f_txt.readline().strip())
        p = np.fromfile(f_bin, 'f', count=2*Ninout)
        return NormalisationLayer(Ninout, p)
    elif layer_name == 'frozen-normalisation':
        Ninout = int(f_txt.readline().strip())
        p = np.fromfile(f_bin, 'f', count=2*Ninout)
        return FrozenNormalisationLayer(Ninout, p)
    elif layer_name == 'dropout':
        Ninout = int(f_txt.readline().strip())
        rate = np.loadtxt(f_txt, max_rows=1)
        return DropoutLayer(Ninout, rate)
    else:
        print('unknown layer type:', layer_name)

class FrozenNormalisationLayer:

    def __init__(self, Ninout, parameters):
        self.Nin = Ninout
        self.Nout = Ninout
        self.alpha = parameters[:Ninout]
        self.beta = parameters[Ninout:]
        self.num_parameters = 0
        self.parameters = np.zeros(0)
        self.keras_parameters = np.zeros(0)

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

class NormalisationLayer:

    def __init__(self, Ninout, parameters):
        self.Nin = Ninout
        self.Nout = Ninout
        self.parameters = parameters.copy()
        self.alpha = self.parameters[:Ninout]
        self.beta = self.parameters[Ninout:]
        self.num_parameters = 2*Ninout
        self.keras_parameters = self.parameters

    def apply(self, x):
        return self.alpha * x + self.beta

    def apply_linearise(self, x):
        self.x = x.copy()
        return self.apply(x)

    def apply_tangent_linear_x(self, dx):
        return self.alpha * dx

    def apply_adjoint_x(self, dy):
        return self.alpha * dy

    def apply_tangent_linear_p(self, dp):
        da = dp[:self.Nin]
        db = dp[self.Nin:]
        return da*self.x + db

    def apply_adjoint_p(self, dy):
        da = self.x * dy
        db = dy
        dp = np.concatenate([da, db])
        return dp

class DropoutLayer:

    def __init__(self, Ninout, rate):
        self.Nin = Ninout
        self.Nout = Ninout
        self.rate = rate
        self.num_parameters = 0
        self.parameters = np.zeros(0)
        self.keras_parameters = np.zeros(0)

    def apply(self, x):
        return x

    def apply_linearise(self, x):
        return self.apply(x)

    def apply_tangent_linear_x(self, dx):
        return dx

    def apply_adjoint_x(self, dy):
        return dy

    def apply_tangent_linear_p(self, dp):
        return np.zeros(self.Nout)

    def apply_adjoint_p(self, dy):
        return np.zeros(0)

class FrozenDenseLayer:

    def __init__(self, Nin, Nout, activation, parameters):
        self.Nin = Nin
        self.Nout = Nout
        self.num_parameters = 0
        self.parameters = np.zeros(0)
        self.keras_parameters = np.zeros(0)
        self.b = parameters[:self.Nout]
        self.w = parameters[self.Nout:].reshape((self.Nout, self.Nin), order='F')
        self.activation = construct_activation(activation)

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
        return np.zeros(self.Nout)

    def apply_adjoint_p(self, dy):
        return np.zeros(0)

class DenseLayer:

    def __init__(self, Nin, Nout, activation, parameters):
        self.Nin = Nin
        self.Nout = Nout
        self.num_parameters = self.Nout * (self.Nin+1)
        self.parameters = parameters.copy()
        self.b = self.parameters[:self.Nout]
        self.w = self.parameters[self.Nout:].reshape((self.Nout, self.Nin), order='F')
        self.activation = construct_activation(activation)

    @property
    def keras_parameters(self):
        b_parameters = self.parameters[:self.Nout]
        w_parameters = self.parameters[self.Nout:]
        return np.concatenate([w_parameters, b_parameters])

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

#--------------------------------------------------
# network
#--------------------------------------------------

class SequentialNetwork:

    def __init__(self):
        self.layers = []

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

    @property
    def keras_parameters(self):
        return self.join_parameters([layer.keras_parameters for layer in self.layers])

    def set_weights_keras(self, keras_model):
        new_w = self.keras_parameters
        new_w_list = []
        index = 0
        for w in keras_model.weights:
            if w.trainable:
                shape = w.numpy().shape
                size = w.numpy().size
                new_w_list.append(new_w[index:index+size].reshape(shape))
                index += size
            else:
                new_w_list.append(w.numpy())
        keras_model.set_weights(new_w_list)

    def split_parameters(self, p):
        p_list = []
        index = 0
        for layer in self.layers:
            p_list.append(p[index:index+layer.num_parameters])
            index += layer.num_parameters
        return p_list

    def join_parameters(self, p_list):
        return np.concatenate(p_list)

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

def fromfile(filename_txt, filename_bin):
    with open(filename_txt, 'r') as f_txt:
        with open(filename_bin, 'rb') as f_bin:
            net_name = f_txt.readline().strip()
            if net_name == 'sequential':
                network = SequentialNetwork()
                num_layers = int(f_txt.readline().strip())
                for i in range(num_layers):
                    layer = layer_fromfile(f_txt, f_bin)
                    network.layers.append(layer)
                return network
            else:
                print('unknown network type:', net_name)

#--------------------------------------------------

