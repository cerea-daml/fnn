
import numpy as np
import tensorflow as tf
from keras_to_fnn import keras_file_to_txt
from pyfnn import fromfile
from tqdm import trange

# set double precision in tensorflow
tf.keras.backend.set_floatx('float64')

def unit_test(Ne):

    Nx = 5
    Ni = 6
    Ny = 4

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]
    gamma = np.random.randn(1)[0]
    delta = np.random.randn(1)[0]

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(Nx,)))
    model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', activation='relu'))
    model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', activation='tanh'))
    model.add(tf.keras.layers.Dense(Ny, bias_initializer='glorot_uniform'))
    model.compile(loss='mse')

    x = np.random.randn(Ne, Nx)
    xn = alpha * x + beta
    y1 = gamma * model.predict(xn) + delta

    fname_1 = 'test_1_model.h5'
    fname_2 = 'test_1_model.txt'

    model.save(fname_1)
    del model
    keras_file_to_txt(fname_2, fname_1, add_norm_in=True, norm_alpha_in=alpha, norm_beta_in=beta,
            add_norm_out=True, norm_alpha_out=gamma, norm_beta_out=delta)

    model = fromfile(fname_2)
    y2 = np.zeros((Ne, Ny))
    for i in range(Ne):
        y2[i] = model.apply(x[i])

    return abs(2*(y1-y2)/(y1+y2)).max()

KEYSIZE = 10
VALUESIZE = 25
PRECISION = 5

def multi_test(Ne, Nt):

    def print_string_line(key, value_a):
        print(f'{key:>{KEYSIZE}} {value_a:>{VALUESIZE}}')

    def print_float_line(key, value_a):
        print(f'{key:>{KEYSIZE}} {value_a:{VALUESIZE}.{PRECISION}f}')

    error = np.array([unit_test(Ne) for _ in trange(Nt, desc='running unit tests')])
    print('-'*100)
    print('test #1')
    print('validation of forward and read of the python toolkit')
    print(f'number of tests = {Nt}')
    print(f'number of points per test = {Ne}')
    print('-'*50)
    print_string_line('test id', 'max error [rel., log10]')
    for (i, e) in enumerate(error):
        print_float_line(i, np.log10(e))
    print('-'*50)
    print_float_line('mean', np.log10(error.mean()))
    print_float_line('std', np.log10(error.std()))
    print('-'*100)

multi_test(100, 10)

