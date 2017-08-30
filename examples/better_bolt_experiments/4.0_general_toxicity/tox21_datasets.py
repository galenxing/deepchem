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


def load_tox21(tasks, featurizer='GraphConv', split='index'):
  """Load Tox21 datasets. Does not do train/test split"""
  # Featurize Tox21 dataset

  print("About to featurize Tox21 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir,
                              "tox21_toxcast_clintox_general.csv.gz")
  if featurizer == 'ECFP':
    featurizer_func = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer_func = dc.feat.ConvMolFeaturizer()
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer_func)
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
      dataset, frac_train=.8, frac_valid=.2, frac_test = 0)
  return (train, valid, test), transformers
