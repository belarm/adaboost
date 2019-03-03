#!/usr/bin/env python3
import adaboost
from functools import reduce
from operator import xor
import numpy as np

def maj(a):
    return np.sum(a) > len(a) / 2



# Our target function shall be an XOR of the features:
split = 200
X, Y = [], []
for i in range(256):
    b = np.unpackbits(np.array([i], dtype=np.uint8))
    X.append(b)
    # Y.append(reduce(xor,b))
    Y.append(maj(b))
# _X, _Y = X, Y
X = np.array(X, dtype=np.int) * 2 - 1
Y = np.array(Y, dtype=np.int) * 2 - 1
indices = np.random.permutation(256)
train_indices = indices[:split]
test_indices = indices[split:]
train_x, test_x = X[train_indices], X[test_indices]
train_y, test_y = Y[train_indices], Y[test_indices]



# print(test_x, test_y)


t = adaboost.AdaBoost()
# t.debug = 2
t.train(train_x, train_y)
# print(t.apply_to_matrix(test_x))
# print(test_y)
# print(t.apply_to_matrix(test_x))
print(f"Final test accuracy: {np.average(np.sign(t.apply_to_matrix(test_x))[:,0] == test_y)}")
# Actually worse than a coin toss!
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(train_x, train_y.reshape((split,)))
print(f"Final GBC test accuracy: {np.average(gbc.predict(test_x) == test_y)}")
