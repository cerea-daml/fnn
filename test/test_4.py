
import numpy as np
import tensorflow as tf
from keras_to_txt import keras_to_txt
from subprocess import run
from networks import fromfile
from scipy.io import FortranFile
from tqdm import trange

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

    fname_1 = 'test_4_model'
    fname_2 = 'test_4_model.txt'
    model.save(fname_1)
    del model
    keras_to_txt(fname_1, fname_2, add_norm_in=True, norm_alpha_in=alpha, norm_beta_in=beta, 
            add_norm_out=True, norm_alpha_out=gamma, norm_beta_out=delta)

    run(['./test_4.x'])

    model = fromfile(fname_2)

    f = FortranFile('test_4_out.bin', 'r')
    x = f.read_reals('f4').reshape(Ne, Nx)
    y1 = f.read_reals('f4').reshape(Ne, Ny)
    dp = f.read_reals('f4')
    dp = model.fortran_to_numpy_parameters(dp)
    dx = f.read_reals('f4').reshape(Ne, Nx)
    dy1 = f.read_reals('f4').reshape(Ne, Ny)
    y2 = np.zeros((Ne, Ny))
    dy2 = np.zeros((Ne, Ny))

    for i in range(Ne):
        y2[i] = model.apply_linearise(x[i])
        dy2[i] = model.apply_tangent_linear(dp, dx[i])

    return max(abs(y1-y2).max(), abs(dy1-dy2).max())

def multi_test(Ne, Nt):
    error = np.array([unit_test(Ne) for _ in trange(Nt, desc='running unit tests')])
    print('-'*100)
    print('test #4')
    print('validation of the tangent linear of the fortran module')
    print(f'number of tests = {Nt}')
    print(f'number of points per test = {Ne}')
    print(f'mean error = {error.mean()}')
    print(f'max error = {error.max()}')
    print('-'*100)

multi_test(100, 10)

