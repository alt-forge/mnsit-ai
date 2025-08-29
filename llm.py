import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[:1000].reshape(1000,28*28) / 255


def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

weights_0_1 = 0.2*np.random.random((784,128)) - 0.1
weights_1_2 = 0.2*np.random.random((128,10)) - 0.1

alpha = 0.01

def one_hot(y, num_classes=10):
    res = np.zeros((y.size, num_classes))
    res[np.arange(y.size), y] = 1
    return res

labels = one_hot(y_train[:1000], 10)

for e in range(350):
    error = 0
    for i in range(len(image)):
        layer_0 = image[i:i+1]
        layer_1 = relu(layer_0.dot(weights_0_1))
        layer_2 = layer_1.dot(weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2)**2)

        layer_2_delta = layer_2 - labels[i:i+1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * drelu(layer_1)

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

    print('Error:', error)