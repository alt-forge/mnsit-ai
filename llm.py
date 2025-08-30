import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


np.random.seed(1)

def relu(x):
    return (x > 0) * x

def drelu(x):
    return (x > 0).astype(float)

weights_0_1 = 0.2*np.random.random((784,100)) - 0.1
weights_1_2 = 0.2*np.random.random((100,10)) - 0.1

alpha = 0.005

def one_hot(y, num_classes=10):
    res = np.zeros((y.size, num_classes))
    res[np.arange(y.size), y] = 1
    return res


image = x_train[:1000].reshape(1000,28*28) / 255
labels = one_hot(y_train[:1000], 10)

test_image = x_test.reshape(len(x_test),28*28) / 255
test_labels = one_hot(y_test, 10)

for e in range(300):
    error = 0
    correct = 0
    for i in range(len(image)):
        layer_0 = image[i:i+1]
        layer_1 = relu(layer_0.dot(weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask*2
        layer_2 = layer_1.dot(weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2)**2)
        correct += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = layer_2 - labels[i:i+1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * drelu(layer_1)

        layer_1_delta *= dropout_mask

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
    if e % 10 == 0:
        test_error = 0
        test_correct = 0
        for i in range(len(test_image)):
            layer_0 = test_image[i:i+1]
            layer_1 = relu(layer_0.dot(weights_0_1))
            layer_2 = layer_1.dot(weights_1_2)
            test_correct += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))
            test_error += np.sum((test_labels[i:i+1] - layer_2)**2)
        sys.stdout.write("\n" + \
                         "Epoch:" + str(e) + \
                         " Test-Error:" + str(test_error/ float(len(test_image)))[0:5] +\
                         " Test-Accuracy:" + str(test_correct/ float(len(test_image)))+\
                         " Train-Error:" + str(error/ float(len(image)))[0:5] +\
                         " Train-Accuracy:" + str(correct/ float(len(image))))