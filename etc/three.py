import numpy as np
from icecream import ic

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    maximum = max(z)
    z = z - maximum
    total = sum(np.exp(z))
    return np.exp(z) / total


x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
#[0.1, 0.3, 0.5]
#[0.2, 0.4, 0.6]
b1 = ([0.1, 0.2, 0.3])

#ic(w1.shape)
a1 = np.dot(x, w1) + b1
ic(a1)
z1 = sigmoid(a1)
ic(z1)

w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])

a2 = np.dot(z1, w2) + b2
ic(a2)
z2 = sigmoid(a2)
ic(z2)

w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])
a3 = np.dot(z2, w3) + b3
z3 = softmax(a3)
ic(z3)


a = np.array([1010, 1000, 990])
ic(softmax(a))
ic(sum(softmax(a))) # softmax関数の合計は１になり、それぞれが確率を示す