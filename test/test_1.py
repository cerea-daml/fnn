
import numpy as np
import tensorflow as tf
from keras_to_txt import keras_to_txt
from networks import fromfile
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

    x = np.random.randn(Ne, Nx)
    xn = alpha * x + beta
    y1 = gamma * model.predict(xn) + delta

    fname_1 = 'test_1_model'
    fname_2 = 'test_1_model.txt'

    model.save(fname_1)
    del model
    keras_to_txt(fname_1, fname_2, add_norm_in=True, norm_alpha_in=alpha, norm_beta_in=beta,
            add_norm_out=True, norm_alpha_out=gamma, norm_beta_out=delta)

    model = fromfile(fname_2)
    y2 = np.zeros((Ne, Ny))
    for i in range(Ne):
        y2[i] = model.apply(x[i])

    return abs(y1-y2).max()

def multi_test(Ne, Nt):
    error = np.array([unit_test(Ne) for _ in trange(Nt, desc='running unit tests')])
    print('-'*100)
    print('test #1')
    print('validation of forward and read of the python toolkit')
    print(f'number of tests = {Nt}')
    print(f'number of points per test = {Ne}')
    print(f'mean error = {error.mean()}')
    print(f'max error = {error.max()}')
    print('-'*100)

multi_test(100, 10)

