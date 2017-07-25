"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import six
import sys

from deepchem.models.tensorgraph import TensorGraph
from deepchem.metrics import to_one_hot

from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Input, GraphConv, BatchNorm, GraphPool, Dense, GraphGather, \
  SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Constant, Weights, Feature, AlphaShare, Add, Multiply, BetaShare, LayerSplitter, SluiceLoss

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from datasets import load_tox21
from record_info import record_info

model_dir = "/tmp/graph_conv"


def graph_conv_model(batch_size, tasks):
  minimizer = 1
  model = TensorGraph(
      model_dir=model_dir, batch_size=batch_size, use_queue=False)
  atom_features = Feature(shape=(None, 75))
  degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
  membership = Feature(shape=(None, ), dtype=tf.int32)

  sluice_loss = []
  deg_adjs = []
  for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)
  gc1 = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[atom_features, degree_slice, membership] + deg_adjs)
  batch_norm1 = BatchNorm(in_layers=[gc1])

  sluice_loss.append(batch_norm1)

  as1 = AlphaShare(in_layers=[batch_norm1, batch_norm1])
  ls1a = LayerSplitter(in_layers=as1, tower_num=0)
  ls1b = LayerSplitter(in_layers=as1, tower_num=1)

  gp1a = GraphPool(in_layers=[ls1a, degree_slice, membership] + deg_adjs)
  gp1b = GraphPool(in_layers=[ls1b, degree_slice, membership] + deg_adjs)

  gc2a = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[gp1a, degree_slice, membership] + deg_adjs)
  gc2b = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[gp1b, degree_slice, membership] + deg_adjs)
  sluice_loss.append(gc2a)
  sluice_loss.append(gc2b)

  as2 = AlphaShare(in_layers=[gc2a, gc2b])
  ls2a = LayerSplitter(in_layers=as2, tower_num=0)
  ls2b = LayerSplitter(in_layers=as2, tower_num=1)

  batch_norm2a = BatchNorm(in_layers=[ls2a])
  batch_norm2b = BatchNorm(in_layers=[ls2b])

  gp2a = GraphPool(
      in_layers=[batch_norm2a, degree_slice, membership] + deg_adjs)
  gp2b = GraphPool(
      in_layers=[batch_norm2b, degree_slice, membership] + deg_adjs)

  densea = Dense(out_channels=128, activation_fn=None, in_layers=[gp2a])
  denseb = Dense(out_channels=128, activation_fn=None, in_layers=[gp2b])

  batch_norm3a = BatchNorm(in_layers=[densea])
  batch_norm3b = BatchNorm(in_layers=[denseb])

  gg1a = GraphGather(
      batch_size=batch_size,
      activation_fn=tf.nn.tanh,
      in_layers=[batch_norm3a, degree_slice, membership] + deg_adjs)
  gg1b = GraphGather(
      batch_size=batch_size,
      activation_fn=tf.nn.tanh,
      in_layers=[batch_norm3b, degree_slice, membership] + deg_adjs)

  costs = []
  labels = []
  count = 0
  for task in tasks:
    if count < len(tasks) / 2:
      classification = Dense(
          out_channels=2, activation_fn=None, in_layers=[gg1a])
      print("first half:")
      print(task)
    else:
      classification = Dense(
          out_channels=2, activation_fn=None, in_layers=[gg1b])
      print('second half')
      print(task)
    count += 1

    softmax = SoftMax(in_layers=[classification])
    model.add_output(softmax)

    label = Label(shape=(None, 2))
    labels.append(label)
    cost = SoftMaxCrossEntropy(in_layers=[label, classification])
    costs.append(cost)

  entropy = Concat(in_layers=costs)
  task_weights = Weights(shape=(None, len(tasks)))
  total_loss = WeightedError(in_layers=[entropy, task_weights])
  #model.set_total_loss(total_loss)
 
  #model.set_sluice_loss(s_cost)
  s_cost = SluiceLoss(in_layers=sluice_loss)
  
  minimizer = Constant(minimizer)
  s_cost = Multiply(in_layers=[minimizer, s_cost])
  new_loss = Add(in_layers=[total_loss, s_cost])
  #model.set_alphas([as1, as2])

  model.set_loss(new_loss)

  def feed_dict_generator(dataset, batch_size, epochs=1):
    for epoch in range(epochs):
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(batch_size, pad_batches=True)):
        d = {}
        for index, label in enumerate(labels):
          d[label] = to_one_hot(y_b[:, index])
        d[task_weights] = w_b
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        d[atom_features] = multiConvMol.get_atom_features()
        d[degree_slice] = multiConvMol.deg_slice
        d[membership] = multiConvMol.membership
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
        yield d

  return model, feed_dict_generator, labels, task_weights


# Load Tox21 dataset
if str(sys.argv[1]) == "all":
  task_arg = tox21_tasks = [
      'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
      'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
  ]
else:
  task_arg = [str(sys.argv[1])]

num_epochs = int(str(sys.argv[2]))
print(str(sys.argv))

tox21_tasks, tox21_datasets, transformers = load_tox21(
    featurizer='GraphConv', split='random', tasks=task_arg)
train_dataset, valid_dataset, test_dataset = tox21_datasets
print(train_dataset.data_dir)
print(valid_dataset.data_dir)
print(test_dataset.data_dir)

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# Batch size of models
batch_size = 100

model, generator, labels, task_weights = graph_conv_model(
    batch_size, tox21_tasks)

model.fit_generator(generator(train_dataset, batch_size, epochs=num_epochs), checkpoint_interval=1000)

print("Evaluating model")
train_scores = model.evaluate_generator(
    generator(train_dataset, batch_size),
    [metric],
    transformers,
    labels,
    #    weights=[task_weights], per_task_metrics = True)
    weights=[task_weights])
valid_scores = model.evaluate_generator(
    generator(valid_dataset, batch_size), [metric],
    transformers,
    labels,
    weights=[task_weights])
#    weights=[task_weights], per_task_metrics = True)
test_scores = model.evaluate_generator(
    generator(test_dataset, batch_size), [metric],
    transformers,
    labels,
    weights=[task_weights])

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

print("Test scores")
print(test_scores)

record_info("tox21_clintox.csv", train_scores, valid_scores, test_scores,
            num_epochs, len(tox21_tasks), task_arg)
