"""
Script that looks at the structure of datasets
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from datasets import load_tox21

#load Tox21 Dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

print("train_dataset")
print(train_dataset.get_shape())
print("valid_dataset")
print(valid_dataset.get_shape())
print("test_dataset")
print(test_dataset.get_shape())

print("X")
print(train_dataset.X)
print("y")
print(train_dataset.y)
print("w")
print(train_dataset.w)
print("ids")
print(train_dataset.ids)

print(train_dataset.get_task_names())
