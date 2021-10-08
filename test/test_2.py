
import numpy as np
from networks import SequentialNetwork

def unit_test(Ne):

    print('-'*100)
    print('test #2')
    print('validation of gradient and adjoint of the python toolkit')

    Nx = 5
    Ni = 6
    Ny = 4

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]
    gamma = np.random.randn(1)[0]
    delta = np.random.randn(1)[0]

    network = SequentialNetwork()
    network.add_layer('normalisation', Nx, alpha, beta)
    network.add_layer('dense', Nx, Ni, activation='tanh', initialisation='randn')
    network.add_layer('dense', Ni, Ni, activation='tanh', initialisation='randn')
    network.add_layer('dense', Ni, Ny, activation='linear', initialisation='randn')
    network.add_layer('normalisation', Ny, gamma, delta)
    network.gradient_test_x(100, 1e-8)
    network.gradient_test_p(100, 1e-8)
    network.adjoint_test_x(100)
    network.adjoint_test_p(100)
    print('-'*100)

unit_test(100)

