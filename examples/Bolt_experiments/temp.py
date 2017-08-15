"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph
from record_info import record_info

tox21_tasks = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

percentages = ['_25', '_50', '_75', '_100']
batch_size = 50


def load_tox21():
  dir1 = 'datasets/tasksplit/tmp/NR-AR_train_25_fingerprint'
  dir2 = 'datasets/tasksplit/tmp/NR-AR_train_50_fingerprint'

  d1 = dc.data.DiskDataset(data_dir=dir1)
  d2 = dc.data.DiskDataset(data_dir=dir2)

  print(d1.y[:20])
  print(d2.y[:20])


load_tox21()
