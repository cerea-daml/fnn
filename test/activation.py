#!/usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np

def construct_activation(name, **kwargs):
    activation_class = dict(
            linear=LinearActivation,
            tanh=TanhActivation,
            relu=ReluActivation,
            )
    return activation_class[name](**kwargs)

class AbstractActivation(ABC):

    def __init__(self, **kwargs):
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

    def __init__(self, **kwargs):
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

    def __init__(self, **kwargs):
        super(TanhActivation, self).__init__(**kwargs)

    def apply(self, z):
        return np.tanh(z)

    def apply_linearise(self, z):
        y = np.tanh(z)
        self.activation_prime = 1 - y**2
        return y

class ReluActivation(AbstractActivation):

    def __init__(self, **kwargs):
        super(ReluActivation, self).__init__(**kwargs)

    def apply(self, z):
        return np.maximum(z, 0)

    def apply_linearise(self, z):
        self.activation_prime = 1 * (z>0)
        return np.maximum(z, 0)

