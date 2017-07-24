"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from pprint import pprint


import os
import numpy as np
import shutil
import deepchem as dc
func = lambda x, k: round((x + k) % 5)
func = np.vectorize(func)


def load_sluice():
    sluice_tasks = ['func1', 'func2']

    X = np.random.rand(10000, 1)
    y1 = np.copy(X)
    y2 = np.copy(X)

    y1 = func(y1, 78)
    y2 = func(y2, 53)
    pprint(X[:20])
    pprint(y1[:20])
    pprint(y2[:20])

    y = np.concatenate((y1, y2), axis=1)

    print(X[:20])
    #temp_X = np.zeros((1000, 10))
    for row, value in enumerate(X):
        break
        temp_X[row, X[row, 0]] = 1

    #X = temp_X
    X_train = X[:8000]
    X_valid = X[8000:9000]
    X_test = X[9000:10000]

    y_train = y[:8000]
    y_valid = y[8000:9000]
    y_test = y[9000:10000]

    train = dc.data.NumpyDataset(X=X_train, y=y_train, n_tasks=2)
    valid = dc.data.NumpyDataset(X=X_valid, y=y_valid, n_tasks=2)
    test = dc.data.NumpyDataset(X=X_test, y=y_test, n_tasks=2)

    print('training data shape')
    print(train.get_shape())

    transformers = []
    return sluice_tasks, (train, valid, test), transformers
