
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

def unit_test(list_rates, Nt):

    def test_one(rate):

        Nx = 5
        Ni = 6
        Ny = 4

        alpha = np.random.randn(Nx)
        beta = np.random.randn(Nx)
        gamma = np.random.randn(Ny)
        delta = np.random.randn(Ny)

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(Nx,)))
        model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', 
            activation='relu'))
        model.add(tf.keras.layers.Dense(Ni, bias_initializer='glorot_uniform', 
            activation='tanh'))
        model.add(tf.keras.layers.Dense(Ny, bias_initializer='glorot_uniform'))
        model.compile(loss='mse')

        fname_1 = 'test_9_model.h5'
        fname_2 = 'test_9_model.txt'
        model.save(fname_1)
        del model
        keras_file_to_txt(fname_2, fname_1, 
                add_norm_in=True, 
                norm_alpha_in=alpha, 
                norm_beta_in=beta, 
                add_norm_out=True, 
                norm_alpha_out=gamma, 
                norm_beta_out=delta, 
                dropout_rates=[rate, rate])

        srun(['./test_9.x'])

        model = fromfile(fname_2)

        Ne = 100
        f = FortranFile('test_9_out.bin', 'r')
        x = f.read_reals(fortran_float).reshape((Ne, Nx))
        y1 = f.read_reals(fortran_float).reshape((Ne, Ny))
        y2 = np.zeros((Ne, Ny))
        f.close()

        for i in range(Ne):
            # this model ignores dropout
            y2[i] = model.apply(x[i])

        return np.sqrt(np.mean(np.square(2*(y1-y2)/(y1+y2))))

    error = np.zeros((len(list_rates), Nt))
    for (j, rate) in enumerate(list_rates):
        for t in trange(Nt, desc=f'testing rate {rate}'):
            error[j, t] = test_one(rate)

    return error

KEYSIZE = 10
VALUESIZE = 25
PRECISION = 5

def multi_test(Nt):

    list_rates = [0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8]
    error = unit_test(list_rates, Nt)

    def print_string_line(key, value_a, value_b):
        print(f'{key:>{KEYSIZE}} {value_a:>{VALUESIZE}} {value_b:>{VALUESIZE}}') 

    def print_float_line(key, value_a, value_b):
        print(f'{key:>{KEYSIZE}} {value_a:{VALUESIZE}.{PRECISION}f} {value_b:{VALUESIZE}.{PRECISION}f}')

    print('-'*100)
    print('test #9')
    print('test of dropout rate')
    print(f'number of tests = {Nt}')
    print(f'number of points per test = 100')
    print('-'*75)
    print_string_line('rate', 'mean error [rel.]', 'std error [rel.]')
    for (i, rate) in enumerate(list_rates):
        print_float_line(rate, error[i].mean(), error[i].std())
    print('-'*100)

multi_test(100)

