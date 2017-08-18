"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from itertools import permutations
np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from hiv_dataset import load_hiv
from gc_sluice_network import graph_conv_sluice
from record_info import record_info
from tox21_datasets import load_tox21


tasks = [['tox21_gen'], ['tox21_gen', 'toxcast_gen'], ['tox21_gen', 'toxcast_gen', 'clintox_gen']]
batch_size = 50
weights = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
sluice_layout = [1, 0, 0, 1, 0, 0, 0, 1]
epoch = 20

for task in tasks:
  tests, transformers = load_tox21(tasks = task, split = 'scaffold')
  for count, train in enumerate(train_datasets):
    for weight in weights:
      for i in range (0,3):
        print(weight)

        metric = dc.metrics.Metric(
            dc.metrics.roc_auc_score, np.mean, mode="classification")
        model = GraphConvTensorGraph(
            num_tasks, batch_size=batch_size, mode='classification')
        model.fit(dataset, nb_epoch=num_epochs)

        print("Evaluating model")
        train_scores = model.evaluate(
            dataset, [metric], transformers=[], per_task_metrics=True)
        valid_scores = model.evaluate(
            valid_dataset, [metric], transformers=[], per_task_metrics=True)

        print("Train scoressss")
        print(train_scores)

        print("Validation scores")
        print(valid_scores)

        record_info(
            file_name='2_random_hiv.csv',
            train=train_scores,
            valid=valid_scores,
            weight=weight,
            epochs=epoch,
            percentage_of_hiv=percentage_of_hiv[count])