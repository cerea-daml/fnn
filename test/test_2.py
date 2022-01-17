
import numpy as np
from pyfnn import SequentialNetwork

def unit_test_gradient(list_eps, Ne):

    def test_one(network, eps):

        network.initialise('randn')
        x = np.random.randn(network.layers[0].Nin)
        y = network.apply_linearise(x)

        dx = np.random.randn(network.layers[0].Nin)
        dp = np.random.randn(network.num_parameters)
        dy = network.apply_tangent_linear(dp, dx)

        network.parameters += eps * dp
        yp = network.apply(x+eps*dx)

        d1 = yp-y
        d2 = eps*dy

        return abs(2*(d1-d2)/(d1+d2)).max()

    Nx = 5
    Ni = 6
    Ny = 4

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]
    gamma = np.random.randn(1)[0]
    delta = np.random.randn(1)[0]

    network = SequentialNetwork()
    network.add_layer('normalisation', Nx, alpha, beta)
    network.add_layer('dense', Nx, Ni, activation='relu', initialisation='randn')
    network.add_layer('dense', Ni, Ni, activation='tanh', initialisation='randn')
    network.add_layer('dense', Ni, Ny, activation='linear', initialisation='randn')
    network.add_layer('normalisation', Ny, gamma, delta)

    error = np.zeros((list_eps.size, Ne))
    for (j, eps) in enumerate(list_eps):
        for i in range(Ne):
            error[j, i] = test_one(network, eps)

    return error

def multi_test_gradient(Ne):

    list_eps = np.power(10, -1.-np.arange(15))
    error = unit_test_gradient(list_eps, Ne) 

    KEYSIZE = 10
    VALUESIZE = 25
    PRECISION = 5

    def print_string_line(key, value_a, value_b):
        print(f'{key:>{KEYSIZE}} {value_a:>{VALUESIZE}} {value_b:>{VALUESIZE}}')

    def print_float_line(key, value_a, value_b):
        print(f'{key:{KEYSIZE}.{PRECISION}f} {value_a:{VALUESIZE}.{PRECISION}f} {value_b:{VALUESIZE}.{PRECISION}f}')

    print('-'*100)
    print('test #2a')
    print('validation of the gradient of the python toolkit')
    print(f'number of points per test = {Ne}')
    print('-'*75)
    print_string_line('eps [log10]', 'mean error [rel, log10]', 'std error [rel, log10]')
    for (i, eps) in enumerate(list_eps):
        print_float_line(np.log10(eps), np.log10(error[i].mean()), np.log10(error[i].std()))
    print('-'*75)

def unit_test_adjoint(Ne):

    def test_one(network):

        network.initialise('randn')
        x = np.random.randn(network.layers[0].Nin)
        y = network.apply_linearise(x)

        dx = np.random.randn(network.layers[0].Nin)
        dp = np.random.randn(network.num_parameters)
        dy = np.random.randn(network.layers[-1].Nout)

        dp_a, dx_a = network.apply_adjoint(dy)
        dy_tl = network.apply_tangent_linear(dp, dx)

        d1 = dp_a @ dp + dx_a @ dx
        d2 = dy_tl @ dy

        return abs(2*(d1-d2)/(d1+d2)).max()

    Nx = 5
    Ni = 6
    Ny = 4

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]
    gamma = np.random.randn(1)[0]
    delta = np.random.randn(1)[0]

    network = SequentialNetwork()
    network.add_layer('normalisation', Nx, alpha, beta)
    network.add_layer('dense', Nx, Ni, activation='relu', initialisation='randn')
    network.add_layer('dense', Ni, Ni, activation='tanh', initialisation='randn')
    network.add_layer('dense', Ni, Ny, activation='linear', initialisation='randn')
    network.add_layer('normalisation', Ny, gamma, delta)

    error = np.zeros((Ne))
    for i in range(Ne):
        error[i] = test_one(network)

    return error

def multi_test_adjoint(Ne):

    error = unit_test_adjoint(Ne) 

    KEYSIZE = 25
    VALUESIZE = 25
    PRECISION = 5

    def print_string_line(key, value_a):
        print(f'{key:>{KEYSIZE}} {value_a:>{VALUESIZE}}')

    def print_float_line(key, value_a):
        print(f'{key:{KEYSIZE}.{PRECISION}f} {value_a:{VALUESIZE}.{PRECISION}f}')

    print('test #2b')
    print('validation of the adjoint of the python toolkit')
    print(f'number of points per test = {Ne}')
    print('-'*75)
    print_string_line('mean error [rel, log10]', 'std error [rel, log10]')
    print_float_line(np.log10(error.mean()), np.log10(error.std()))
    print('-'*100)

multi_test_gradient(100)
multi_test_adjoint(100)

