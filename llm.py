import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


np.random.seed(1)

def relu(x):
    return (x > 0) * x

def drelu(x):
    return (x > 0).astype(float)

weights_0_1 = 0.2*np.random.random((784,128)) - 0.1
weights_1_2 = 0.2*np.random.random((128,10)) - 0.1

alpha = 0.001

def one_hot(y, num_classes=10):
    res = np.zeros((y.size, num_classes))
    res[np.arange(y.size), y] = 1
    return res

image = x_train[:10000].reshape(10000,28*28) / 255
labels = one_hot(y_train[:10000], 10)

val_image = x_test[:100].reshape(100,28*28) / 255
val_labels = one_hot(y_test[:100], 10)

for e in range(20):
    error = 0
    correct = 0
    for i in range(len(image)):
        layer_0 = image[i:i+1]
        layer_1 = relu(layer_0.dot(weights_0_1))
        layer_2 = layer_1.dot(weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2)**2)
        correct += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = layer_2 - labels[i:i+1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * drelu(layer_1)

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
    #sys.stdout.write("\rError: " + str(float(error/len(image))) + " Accuracy: " + str(correct) + "/" + str(len(image)))
correct = 0
for i in range(len(val_image)):
    layer_0 = val_image[i:i+1]
    layer_1 = relu(layer_0.dot(weights_0_1))
    layer_2 = layer_1.dot(weights_1_2)
    correct += int(np.argmax(layer_2) == np.argmax(val_labels[i:i+1]))

print("Accuracy: " + str(correct) + "/" + str(len(val_image)))