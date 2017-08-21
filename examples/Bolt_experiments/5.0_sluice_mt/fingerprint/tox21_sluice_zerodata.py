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
from gc_sluice_network import graph_conv_sluice
from tox21_datasets import load_tox21
from record_info import record_info

tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

sluice_layout = [1, 0, 0, 1, 0, 0, 0, 1]

epoch = 15
batch_size = 50

for task in tox21_tasks:
  tox21_tasks, tox21_datasets, transformers = load_tox21(tasks = [task],
      featurizer='GraphConv', split='fingerprint')
  for i in range(0,3):
    
    train_dataset, valid_dataset, test_dataset = tox21_datasets

    print(train_dataset.data_dir)
    print(valid_dataset.data_dir)

    metric = dc.metrics.Metric(
            dc.metrics.roc_auc_score, np.mean, mode="classification")
        # Batch size of models

    model, generator, labels, task_weights = graph_conv_sluice(
        n_tasks=len(tox21_tasks),
        batch_size=batch_size,
        mode='classification',
        minimizer=1,
        sluice_layout=sluice_layout,
        tensorboard=True)

    model.fit_generator(generator(train_dataset, batch_size, epochs=epoch))

    print("Evaluating model")
    train_scores = model.evaluate_generator(
        generator(train_dataset, batch_size), [metric],
        transformers,
        labels,
        weights=[task_weights],
        per_task_metrics = True)

    valid_scores = model.evaluate_generator(
        generator(valid_dataset, batch_size), [metric],
        transformers,
        labels,
        weights=[task_weights],
        per_task_metrics = True)

    print("Train scores")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)
    record_info(file_name= 'tox21_sluice_baseline_fingerprint.csv', train= train_scores, valid = valid_scores, task = task)

for i in range(0,3):
  tox21_tasks, tox21_datasets, transformers = load_tox21(tasks = tox21_tasks,
    featurizer='GraphConv', split='fingerprint')
  train_dataset, valid_dataset, test_dataset = tox21_datasets

  print(train_dataset.data_dir)
  print(valid_dataset.data_dir)

  metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, np.mean, mode="classification")
      # Batch size of models

  model, generator, labels, task_weights = graph_conv_sluice(
      n_tasks=len(tox21_tasks),
      batch_size=batch_size,
      mode='classification',
      minimizer=1,
      sluice_layout=sluice_layout,
      tensorboard=True)

  model.fit_generator(generator(train_dataset, batch_size, epochs=epoch))

  print("Evaluating model")
  train_scores = model.evaluate_generator(
      generator(train_dataset, batch_size), [metric],
      transformers,
      labels,
      weights=[task_weights],
      per_task_metrics = True)

  valid_scores = model.evaluate_generator(
      generator(valid_dataset, batch_size), [metric],
      transformers,
      labels,
      weights=[task_weights],
      per_task_metrics = True)

  print("Train scores")
  print(train_scores)

  print("Validation scores")
  print(valid_scores)
  record_info(file_name= 'tox21_sluice_baseline_fingerprint.csv', train= train_scores, valid = valid_scores, task = task)

