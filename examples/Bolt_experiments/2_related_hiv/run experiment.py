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


def addHIVdata(tox21_dataset, HIV_dataset, tox21_valid):
  hiv_X = HIV_dataset.X
  hiv_y = HIV_dataset.y
  hiv_w = HIV_dataset.w
  hiv_ids = HIV_dataset.ids

  tox21_X = tox21_dataset.X
  tox21_y = tox21_dataset.y
  tox21_w = tox21_dataset.w
  tox21_ids = tox21_dataset.ids

  tox21 = [tox21_X, tox21_y, tox21_w, tox21_ids]

  nrows = tox21_X.shape[0]
  hiv_variables = [hiv_X, hiv_y, hiv_w, hiv_ids]
  hiv25, hiv50, hiv75, hiv100 = [], [], [], []
  for var in hiv_variables:
    hiv25.append(var[:int(nrows * .25)])
    hiv50.append(var[:int(nrows * .5)])
    hiv75.append(var[:int(nrows * .75)])
    hiv100.append(var)

  hiv_dataset_percentages = [hiv25, hiv50, hiv75, hiv100]

  combined = []

  for dataset in hiv_dataset_percentages:
    for count, var in enumerate(dataset):
      if count == 0 or count == 3:
        print(tox21[count].shape)
        print(var.shape)
        temp1 = np.concatenate((tox21[count], var))
        combined.append(temp1)
      if count == 1 or count == 2:
        temp1 = np.zeros((var.shape[0], tox21[count].shape[1]))
        temp2 = np.zeros((tox21[count].shape[0], var.shape[1]))
        temp1 = np.concatenate((tox21[count], temp1))
        temp2 = np.concatenate((temp2, var))
        combined.append(np.concatenate((temp1, temp2), axis=1))

  print(len(combined))

  i = 0
  train_datasets = []
  for count in range(0, 16, 4):
    print(combined[count])
    print(combined[count + 1])
    print(combined[count + 2])
    print(combined[count + 3])
    train_datasets.append(
        dc.data.DiskDataset.from_numpy(combined[count], combined[
            count + 1], combined[count + 2], combined[count + 3]))

  print(len(train_datasets))

  for dataset in train_datasets:
    print(dataset.X.shape)
    print(dataset.y.shape)
    print(dataset.w.shape)
    print(dataset.ids.shape)

  valid_y = tox21_valid.y
  valid_w = tox21_valid.w

  temp = np.zeros((valid_y.shape[0], 1))
  valid_y = np.concatenate((valid_y, temp), axis =1)
  valid_w = np.concatenate((valid_w, temp), axis =1)

  valid = dc.data.DiskDataset.from_numpy(tox21_valid.X, valid_y, valid_w, tox21_valid.ids)

  return train_datasets, valid


tox21_train = dc.data.DiskDataset(
    data_dir='../post_split/tox21_fingerprint_train')
valid_dataset = dc.data.DiskDataset(
    data_dir='../post_split/tox21_fingerprint_valid')
hiv_train = dc.data.DiskDataset(data_dir='hiv_random_train')

train_datasets, valid_dataset = addHIVdata(tox21_train, hiv_train, valid_dataset)

batch_size = 50
weights = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
sluice_layout = [1, 0, 0, 1, 0, 0, 0, 1]
epoch = 20
percentage_of_hiv = [0.25, 0.5, 0.75, 1]
for count, train in enumerate(train_datasets):
  for weight in weights:
    print(weight)
    # Load Tox21 dataset
    # Fit models
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")

    # Batch size of models
    model, generator, labels, task_weights = graph_conv_sluice(
        n_tasks=13,
        batch_size=batch_size,
        mode='classification',
        minimizer=weight,
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
        generator(valid_dataset, batch_size),
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
        file_name='2_random_hiv.csv',
        train=train_scores,
        valid=valid_scores,
        weight=weight,
        epochs=epoch,
        percentage_of_hiv=percentage_of_hiv[count])
