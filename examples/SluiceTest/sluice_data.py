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
func = lambda x, k: round((x + k))
func = np.vectorize(func)


def load_sluice():
    sluice_tasks = ['func1', 'func2']

    
    X_train = np.random.randint(2000, size=(80000, 1))
    X_test = np.random.randint(2000, high=3000, size=(10000, 1))
    X_valid = np.random.randint(1000, high=2000, size=(10000, 1))
    """

    X_train = np.random.randint(8, size=(80000, 1))
    X_test = np.random.randint(0, high=8, size=(10000, 1))
    X_valid = np.random.randint(0, high=8, size=(10000, 1))
    """

    datasets = [X_train, X_test, X_valid]

    task1 = []
    task2 = []

    for dataset in datasets:
        y1 = y2 = np.copy(dataset)
        task1.append(func(y1, 10))
        task2.append(func(y2, 12))

    y = []
    for count, task in enumerate(task1):
        y.append(np.concatenate((task1[count], task2[count]), axis =1))

    y_train = y[0]
    y_valid = y[2]
    y_test = y[1]

    print(len(y_train))
    print(len(y_valid))
    print(len(y_test))

    train = dc.data.NumpyDataset(X=X_train, y=y_train, n_tasks=2)
    valid = dc.data.NumpyDataset(X=X_valid, y=y_valid, n_tasks=2)
    test = dc.data.NumpyDataset(X=X_test, y=y_test, n_tasks=2)

    print('training data shape')
    print(train.get_shape())

    transformers = []
    return sluice_tasks, (train, valid, test), transformers
