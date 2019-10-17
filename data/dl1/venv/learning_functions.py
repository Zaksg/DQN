import numpy as np
import math
from sklearn.model_selection import train_test_split

EPSILON = 0.03

def initialize_parameters(layer_dims):
    return_dict = {}
    for i, (dim, ndim) in enumerate(zip(layer_dims, layer_dims[1:])):
        i = i + 1
        return_dict['W' + str(i)] = np.random.randn(ndim, dim) / np.sqrt(dim)
        return_dict['b' + str(i)] = np.zeros((ndim, 1))
    return return_dict


def linear_forward(A, W, b):
    linear_cache = {
        'A': A,
        'W': W,
        'b': b
    }
    Z = np.dot(W, A) + b
    return Z, linear_cache

def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A, z

def relu(Z):
    return np.maximum(0, Z), Z

def linear_activation_forward(A_prev, W, B, activation='relu'):
    Z, linear_cache = linear_forward(A_prev, W, B)
    activation_func = {
        'softmax': softmax,
        'relu': relu
    }
    A, activation_cache = activation_func[activation](Z)
    cache = {'linear_cache': linear_cache,
             'activation_cache': activation_cache}
    return A, cache

def dropout_func(a, keep_prob):
    d = np.random.rand(a.shape[0], a.shape[1])
    d[d < keep_prob] = 0
    d[d >= keep_prob] = 1. / (1-keep_prob)
    a = np.multiply(a, d)
    return a


def L_model_forward(X, parameters, use_batchnorm=False, dropout=False):
    caches = []
    A = X
    L = len(parameters) // 2
    for i in range(1, L):
        A_prev = A
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        if isinstance(A_prev, tuple):
            print('a')
        A, cache = linear_activation_forward(A_prev, W, b, activation='relu')
        A = dropout_func(A, dropout) if dropout else A
        A = apply_batchnorm(A) if use_batchnorm else A

        caches.append(cache)
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation='softmax')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(AL + 1e-15))
    cost = np.squeeze(cost)
    return cost


def apply_batchnorm(A):
    mean = np.mean(A)
    varience = np.var(A)
    A_norm = (A - mean) / np.sqrt(varience + 1e-7)
    return A_norm



def Linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']
    b = cache['b']
    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation='relu'):
    if activation == 'softmax':
        dZ = softmax_backward(dA, cache['activation_cache'])
        dA_prev, dW, db = Linear_backward(dZ, cache['linear_cache'])
    else:
        dZ = relu_backward(dA, cache['activation_cache'])
        dA_prev, dW, db = Linear_backward(dZ, cache['linear_cache'])


    return dA_prev, dW, db


def softmax_backward(dA, activation_cache):
    z = activation_cache
    dZ = dA - softmax(z)[0]
    return dZ


def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = AL - Y
    cache = caches[-1]
    linear_cache = cache['linear_cache']
    dA_prev, dW, db = Linear_backward(dAL, linear_cache)
    grads['dA' + str(L)] = dA_prev
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db

    for i in reversed(range(L - 1)):
        cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(i + 2)], cache, activation="relu")
        grads['dA' + str(i+1)] = dA_prev_temp
        grads['dW' + str(i+1)] = dW_temp
        grads['db' + str(i+1)] = db_temp
    return grads


def Update_parameters(parameters, grads, learning_rate=0.001):
    return_params = {}
    L = round(len(parameters) / 2)
    for i in range(1, L + 1):
        currentDW = grads['dW' + str(i)]
        currentDb = grads['db' + str(i)]
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        W = W - learning_rate * currentDW
        b = b - learning_rate * currentDb
        return_params['W' + str(i)] = W
        return_params['b' + str(i)] = b
    return return_params


def training_validation_split(X, Y):
    x_train_data = X.reshape((X.shape[0], 784))
    x_train, x_val, y_train, y_val = train_test_split(x_train_data, Y.T, test_size=0.25)
    return x_train.T, y_train.T, x_val.T, y_val.T

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, batch_size,
                  use_batchnorm=False):
    training_steps = 0
    prev_accuracy = 0
    costs = []
    stop = False
    accuracy_change = 0
    parameters = initialize_parameters(layer_dims)
    for i in range(num_iterations):
        if stop:
            break
        x_train, y_train, x_val, y_val = training_validation_split(X, Y)
        print("num of iteration: " + str(i))
        minibatches = batch_split(x_train, y_train, batch_size)
        for x, y in minibatches:
            training_steps += 1
            AL, caches = L_model_forward(x, parameters, use_batchnorm)
            grads = L_model_backward(AL, y, caches)
            parameters = Update_parameters(parameters, grads, learning_rate)
            if training_steps % 100 == 0:
                cost = compute_cost(AL, y)
                costs.append(cost)
                accuracy = Predict(x_val, y_val, parameters, use_batchnorm=use_batchnorm) * 100

                print('Iter: {}, Training Steps {}: {}'.format(i, training_steps, cost))
                print('Validation accuracy: {}'.format(accuracy))
                if abs(accuracy - prev_accuracy) <= EPSILON:
                    accuracy_change += 1
                else:
                    accuracy_change = 0
                if accuracy_change == 3:
                    stop = True
                prev_accuracy = accuracy

    return parameters, costs, training_steps, accuracy, i


def Predict(X, Y, parameters, use_batchnorm=False):
    AL, caches = L_model_forward(X, parameters, use_batchnorm)
    m = Y.shape[1]
    Y = np.argmax(Y, axis=0)
    predictions = np.argmax(AL, axis=0)
    counter = (Y == predictions).sum()
    accuracy = counter / m
    return accuracy


def batch_split(X, Y, mini_batch_size=64):
    randomize = np.arange(X.shape[1])
    np.random.shuffle(randomize)
    shuffled_X = X[:, randomize]
    shuffled_Y = Y[:, randomize]
    mini_batches =[]
    m = X.shape[1]
    full_size_batch = math.floor(m / mini_batch_size)
    for i in range(0, X.shape[1], mini_batch_size):
        X_train_mini = shuffled_X[:, i:i + mini_batch_size]
        y_train_mini = shuffled_Y[:, i:i + mini_batch_size]
        mini_batches.append((X_train_mini, y_train_mini))
    if m % mini_batch_size != 0:
        X_train_mini = shuffled_X[:, full_size_batch * mini_batch_size:]
        y_train_mini = shuffled_Y[:, full_size_batch * mini_batch_size:]
        mini_batches.append((X_train_mini, y_train_mini))

    return mini_batches