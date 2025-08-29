'''
def vector_sum(vec_a):
    return sum(vec_a)
def vector_average(vec_a):
    return sum(vec_a)/len(vec_a)
def elementwise_addition(vec_a,vec_b):
    vec_sum = []
    for i in range(0, len(vec_a)):
        vec_sum.append(vec_a[i]+vec_b[i])
    return vec_sum
def elementwise_multiplication(vec_a,vec_b):
    vec_sum = []
    for i in range(0, len(vec_a)):
        vec_sum.append(vec_a[i]*vec_b[i])
    return vec_sum
a = [.5,0,.5,0]
b = [1,1,0,0]

def scalar_multiplication(vec_a,vec_b):
    return vector_sum(elementwise_multiplication(a,b))

def w_sum(a,b):
    assert(len(a)==len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i]*b[i])
    return output

def ele_mul(number,vector):
    output = [0,0,0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output
weights = [0.3,0.2,0.9]
def neural_network(input,weights):
    pred = ele_mul(input,weights)
    return pred
input = 0.65
print(neural_network(input,weights))
'''

import numpy as np
a = np.zeros((1,4))
b = np.zeros((4,3))
print(a)
print(b)