
import numpy as np
import tensorflow as tf
from keras_to_fnn import keras_file_to_txt
from subprocess import run as srun
from pyfnn import fromfile
from tqdm import trange

# set double precision in tensorflow
tf.keras.backend.set_floatx('float64')

def unit_test(Ne):

    Nx = 5
    Ni = 6
    Ny = 4

    alpha = np.random.randn(Nx)
    beta = np.random.randn(Nx)
    gamma = np.random.randn(Ny)
    delta = np.random.randn(Ny)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(Nx,)))
    model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', activation='relu'))
    model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', activation='tanh'))
    model.add(tf.keras.layers.Dense(Ny, bias_initializer='glorot_uniform'))
    model.layers[1].trainable = False
    model.compile(loss='mse')

    fname_1 = 'test_3_model.keras'
    fname_2 = 'test_3_model.txt'
    fname_3 = 'test_3_model.bin'

    model.save(fname_1)
    del model
    keras_file_to_txt(
        fname_2, 
        fname_3,
        fname_1, 
        norm_in=dict(
            freeze=True,
            alpha=alpha,
            beta=beta,
        ),
        norm_out=dict(
            freeze=False,
            alpha=gamma,
            beta=delta,
        ),
    )
    srun(['./test_3.x'])

    model = fromfile(fname_2, fname_3)

    with open('test_3_out.bin', 'rb') as f:
        x = np.fromfile(f, count=Ne*Nx).reshape((Ne, Nx))
        y1 = np.fromfile(f, count=Ne*Ny).reshape((Ne, Ny))
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
    print('test #3')
    print('validation of forward and read of the fortran module')
    print(f'number of tests = {Nt}')
    print(f'number of points per test = {Ne}')
    print('-'*50)
    print_string_line('test id', 'max error [rel.]')
    for (i, e) in enumerate(error):
        print_float_line(i, e)
    print('-'*50)
    print_float_line('mean', error.mean())
    print_float_line('std', error.std())
    print('-'*100)

multi_test(100, 10)

