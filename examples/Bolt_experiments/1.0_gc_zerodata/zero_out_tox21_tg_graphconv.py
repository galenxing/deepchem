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
num_epochs = 10


def load_tox21(task_name, num_tasks):
  """returns 4 train_datasets as tuples and a valid dataset for 
    the task_name and also zeros out data to match num_tasks
    """
  train_dirs = []
  train_datasets = []
  task_index = tox21_tasks.index(task_name)

  #make all the datasets
  for percentage in percentages:
    tmp_dir = 'datasets/tasksplit/postfp/' + \
        task_name + '_train' + percentage + '_fingerprint'
    print(tmp_dir)
    tmp_dataset = dc.data.DiskDataset(data_dir=tmp_dir)
    train_datasets.append(tmp_dataset)

  #print(train_datasets[0].get_shape())
  #print(train_datasets[0].y[:20])
  #print(train_datasets[1].y[:20])

  range_tasks = []
  range_tasks.append(task_index)
  for i in range(1, num_tasks):
    task_index += 1
    if task_index == 12:
      task_index = 0
    print(task_index)
    range_tasks.append(task_index)

  for count, dataset in enumerate(train_datasets):
    test = dataset
    X = dataset.X
    y = dataset.y[:, range_tasks]
    w = dataset.w[:, range_tasks]
    ids = dataset.ids
    print(y.shape)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    print([tox21_tasks[x] for x in range_tasks])
    dataset.selectedTasks = [tox21_tasks[x] for x in range_tasks]

    dataset.percentage = percentages[count]
    train_datasets[count] = dataset
    print(dataset.percentage)
    print(dataset.get_shape())

  # print(train_datasets)
  valid_data_dir = 'datasets/tasksplit/postfp/' + task_name + '_validfingerprint'
  valid_dataset = dc.data.DiskDataset(valid_data_dir)
  X = valid_dataset.X
  y = valid_dataset.y[:, range_tasks]
  w = valid_dataset.w[:, range_tasks]
  ids = valid_dataset.ids
  valid_dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
  # print(valid_dataset.get_shape())

  return tuple(train_datasets), valid_dataset


def run_model(dataset, valid_dataset, num_tasks):

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

  return train_scores, valid_scores


def run_model_three_times(dataset, valid_dataset, num_tasks, task):
  filename = task + '.csv'
  for i in range(0, 3):
    train_scores, valid_scores = run_model(dataset, valid_dataset, num_tasks)
    record_info(
        file_name=filename,
        train=train_scores,
        valid=valid_scores,
        epochs=num_epochs,
        percentage=dataset.percentage,
        tasks=dataset.selectedTasks)


#testing = True
# if testing == True:
model_dir = "tmp/graph_conv"
# elif testing == False:
#  model_dir = "/scr/xing/tmp/graph_conv"

list_of_num_tasks = [1, 4, 8, 12]

for task in tox21_tasks:
  for num_tasks in list_of_num_tasks:
    train_datasets, valid_dataset = load_tox21(task, num_tasks)
    for dataset in train_datasets:
      run_model_three_times(dataset, valid_dataset, num_tasks, task)
