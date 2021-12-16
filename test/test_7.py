
import numpy as np
import tensorflow as tf
from keras_to_fnn import keras_file_to_txt
from subprocess import run as srun
from pyfnn import fromfile
from scipy.io import FortranFile
from tqdm import trange

# set double precision in tensorflow
tf.keras.backend.set_floatx('float64')

# use double format in fortan
fortran_float = 'f8'

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
    model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', activation='tanh'))
    model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', activation='tanh'))
    model.add(tf.keras.layers.Dense(Ny, bias_initializer='glorot_uniform'))
    model.compile(loss='mse')

    fname_1 = 'test_7_model.h5'
    fname_2 = 'test_7_model_in.txt'
    fname_3 = 'test_7_model_out.txt'
    model.save(fname_1)
    del model
    keras_file_to_txt(fname_2, fname_1, add_norm_in=True, norm_alpha_in=alpha, norm_beta_in=beta, 
            add_norm_out=True, norm_alpha_out=gamma, norm_beta_out=delta)

    srun(['./test_7.x'])

    model = fromfile(fname_3)

    f = FortranFile('test_7_out.bin', 'r')
    x = f.read_reals(fortran_float).reshape(Ne, Nx)
    y1 = f.read_reals(fortran_float).reshape(Ne, Ny)
    y2 = np.zeros((Ne, Ny))
    f.close()

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
    print('test #7')
    print('validation of parameter replacement in the fortran module')
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

