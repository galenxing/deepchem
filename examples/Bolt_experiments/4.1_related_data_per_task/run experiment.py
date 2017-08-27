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

import os
import shutil

tf.set_random_seed(123)
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph

from record_info import record_info

batch_size = 50
sluice_layout = [1, 0, 0, 1, 0, 0, 0, 1]
epoch = 20


def load_toxcast(split='scaffold'):
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, "toxcast_data.csv.gz")
  dataset = dc.utils.save.load_from_disk(dataset_file)
  featurizer_func = dc.feat.ConvMolFeaturizer()
  toxcast_tasks = dataset.columns.values[1:].tolist()

  loader = dc.data.CSVLoader(
      tasks=toxcast_tasks, smiles_field="smiles", featurizer=featurizer_func)

  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
      'butina': dc.splits.ButinaSplitter(),
      'fingerprint': dc.splits.FingerprintSplitter()
  }

  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(
      dataset, frac_train=1, frac_valid=0, frac_test=0)
  return train, toxcast_tasks


def load_clintox(featurizer='GraphConv', split='scaffold'):
  """Load clintox datasets."""

  # Load clintox dataset
  print("About to load clintox dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, "clintox.csv.gz")
  dataset = dc.utils.save.load_from_disk(dataset_file)
  clintox_tasks = dataset.columns.values[1:].tolist()
  print("Tasks in dataset: %s" % (clintox_tasks))
  print("Number of tasks in dataset: %s" % str(len(clintox_tasks)))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize clintox dataset
  print("About to featurize clintox dataset.")
  featurizers = {
      'ECFP': dc.feat.CircularFingerprint(size=1024),
      'GraphConv': dc.feat.ConvMolFeaturizer()
  }
  featurizer = featurizers[featurizer]
  loader = dc.data.CSVLoader(
      tasks=clintox_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Transform clintox dataset
  print("About to transform clintox dataset.")
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  # Split clintox dataset
  print("About to split clintox dataset.")
  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(
      dataset, frac_train=1, frac_valid=0, frac_test=0)

  return train, clintox_tasks


def combine_datasets(dataset1, dataset2):
  d1_X = dataset1.X
  d1_y = dataset1.y
  d1_w = dataset1.w
  d1_ids = dataset1.ids

  d2_X = dataset2.X
  d2_y = dataset2.y
  d2_w = dataset2.w
  d2_ids = dataset2.ids

  d1 = [d1_X, d1_y, d1_w, d1_ids]
  d2 = [d2_X, d2_y, d2_w, d2_ids]

  nrows = d2_X.shape[0]

  combined = []
  for count, var in enumerate(d1):
    if count == 0 or count == 3:
      temp1 = np.concatenate((d2[count], var))
      combined.append(temp1)
    if count == 1 or count == 2:
      temp1 = np.zeros((var.shape[0], d2[count].shape[1]))
      temp2 = np.zeros((d2[count].shape[0], var.shape[1]))
      temp1 = np.concatenate((d2[count], temp1))
      temp2 = np.concatenate((temp2, var))
      combined.append(np.concatenate((temp1, temp2), axis=1))

  combined_dataset = dc.data.DiskDataset.from_numpy(combined[0], combined[1],
                                                    combined[2], combined[3])

  print(combined_dataset.X.shape)
  print(combined_dataset.y.shape)
  print(combined_dataset.w.shape)
  print(combined_dataset.ids.shape)

  return combined_dataset


def add_to_valid(dataset, tasks):
  valid_y = dataset.y
  valid_w = dataset.w

  temp = np.zeros((valid_y.shape[0], len(tasks)))
  valid_y = np.concatenate((valid_y, temp), axis=1)
  valid_w = np.concatenate((valid_w, temp), axis=1)

  valid = dc.data.DiskDataset.from_numpy(dataset.X, valid_y, valid_w,
                                         dataset.ids)
  return valid


def load_data(toxcast=False, clintox=False):
  tox21_train = dc.data.DiskDataset(data_dir='tox21_fingerprint_train')
  tox21_valid = dc.data.DiskDataset(data_dir='tox21_fingerprint_valid')
  tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
           'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
           'SR-HSE', 'SR-MMP', 'SR-p53']
  if toxcast:
    toxcast_dataset, toxcast_tasks = load_toxcast()
    tox21_train = combine_datasets(tox21_train, toxcast_dataset)
    tox21_valid = add_to_valid(tox21_valid, toxcast_tasks)
    tasks += toxcast_tasks
  if clintox:
    clintox_dataset, clintox_tasks = load_clintox()
    tox21_train = combine_datasets(tox21_train, clintox_dataset)
    tox21_valid = add_to_valid(tox21_valid, clintox_tasks)
    tasks += clintox_tasks



  return tox21_train, tox21_valid, tasks 


train, test, tasks= load_data(toxcast=True, clintox=True)
print(train.X.shape)
print(train.y.shape)
print(train.w.shape)
print(train.ids.shape)

print(test.X.shape)
print(test.y.shape)
print(test.w.shape)
print(test.ids.shape)

print(train.data_dir)
print(test.data_dir)

epochs = 20

for i in range(0, 3):
  metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
  model = GraphConvTensorGraph(
      n_tasks=len(tasks), batch_size=batch_size, mode='classification')
  model.fit(train, nb_epoch=epochs)

  print("Evaluating model")
  train_scores = model.evaluate(
      train, [metric], transformers=[], per_task_metrics=True)
  valid_scores = model.evaluate(
      test, [metric], transformers=[], per_task_metrics=True)

  print("Train scoressss")
  print(train_scores)

  print("Validation scores")
  print(valid_scores)

  record_info(
      file_name='4.1_related_data_per_task.csv',
      train=train_scores,
      valid=valid_scores,
      weight='',
      epochs=epoch,
      architecture='gc')
  metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")