"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc


def load_tox21(task, featurizer='ECFP', split='index', testing=True):
  """Load Tox21 datasets. Does not do train/test split"""
  # Featurize Tox21 dataset
  dataset_path = 'datasets/tasksplit/tox21_' + task + '_1.0.csv.gz'
  save_dir = 'datasets/tasksplit/tmp/' + task

  tox21_tasks = [
      'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
      'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
  ]

  print("About to featurize Tox21 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, dataset_path)

  if featurizer == 'ECFP':
    featurizer_func = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer_func = dc.feat.ConvMolFeaturizer()
  loader = dc.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer_func)
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
  train, valid, test = splitter.train_valid_fraction_split(
      dataset,
      frac_train=.8,
      frac_valid=.2,
      frac_test=0,
      train_dir=save_dir + '_train',
      valid_dir=save_dir + '_valid',
      test_dir=save_dir + '_test')
  return (train, valid, test), transformers
