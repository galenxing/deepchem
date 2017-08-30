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
from tox21_datasets import load_tox21
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph
from record_info import record_info

testing = True

if testing == True:
  model_dir = "tmp/graph_conv"
elif testing == False:
  model_dir = "/scr/xing/tmp/graph_conv"

# Load Tox21 dataset

tox21_tasks = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

percentage = ['_25', '_50', '_75', '_100']

for task in tox21_tasks:
  tox21_datasets, transformers = load_tox21(
      task=task, featurizer='GraphConv', split='random')
  train_datasets, valid_dataset, test_dataset = tox21_datasets
  count = 0
  for dataset in train_datasets:
    filename = task + percentage[count] + "_12tasks.csv"
    count += 1
    for i in range(3):
      # Fit models
      metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, np.mean, mode="classification")

      # Batch size of models
      batch_size = 50

      model = GraphConvTensorGraph(
          len(tox21_tasks), batch_size=batch_size, mode='classification')

      model.fit(dataset, nb_epoch=10)

      print("Evaluating model")
      train_scores = model.evaluate(
          dataset, [metric], transformers, per_task_metrics=True)
      valid_scores = model.evaluate(
          valid_dataset, [metric], transformers, per_task_metrics=True)

      print("Train scoressss")
      print(train_scores)

      print("Validation scores")
      print(valid_scores)

      #record_info(
      #file_name=filename, train=train_scores, valid=valid_scores, epochs=10)
