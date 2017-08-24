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
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph

from hiv_dataset import load_hiv
from gc_sluice_network import graph_conv_sluice
from record_info import record_info
from tox21_datasets import load_tox21

tasks = [['tox21_gen'], ['tox21_gen', 'toxcast_gen'],
         ['tox21_gen', 'toxcast_gen', 'clintox_gen']]
batch_size = 50
sluice_layout = [1, 0, 0, 1, 0, 0, 0, 1]
epoch = 20





for task in tasks:
  datasets, transformers = load_tox21(tasks=task, split='scaffold')
  train, valid, test = datasets
  for i in range(0, 3):
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
    model = GraphConvTensorGraph(
        n_tasks=len(task), batch_size=batch_size, mode='classification')
    model.fit(train, nb_epoch=epoch)

    print("Evaluating model")
    train_scores = model.evaluate(
        train, [metric], transformers=[], per_task_metrics=True)
    valid_scores = model.evaluate(
        valid, [metric], transformers=[], per_task_metrics=True)

    print("Train scoressss")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)

    record_info(
        file_name='4.0_general_tox.csv',
        train=train_scores,
        valid=valid_scores,
        weight='',
        epochs=epoch,
        architecture='gc')
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")

    # Batch size of models
    model, generator, labels, task_weights = graph_conv_sluice(
        n_tasks=len(task),
        batch_size=batch_size,
        mode='classification',
        minimizer=0.25,
        sluice_layout=sluice_layout,
        tensorboard=True)

    model.fit_generator(generator(train, batch_size, epochs=epoch))

    print("Evaluating model")
    train_scores = model.evaluate_generator(
        generator(train, batch_size),
        metrics=[metric],
        transformers=[],
        labels=labels,
        weights=[task_weights],
        per_task_metrics=True)

    valid_scores = model.evaluate_generator(
        generator(valid, batch_size),
        metrics=[metric],
        transformers=[],
        labels=labels,
        weights=[task_weights],
        per_task_metrics=True)

    print("Train scores")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)

    record_info(
        file_name='4.0_general_tox.csv',
        train=train_scores,
        valid=valid_scores,
        weight=0.25,
        epochs=epoch,
        architecture='sluice')