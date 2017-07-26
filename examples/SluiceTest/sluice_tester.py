"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import six

from deepchem.models.tensorgraph import TensorGraph
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Input, GraphConv, Add, SluiceLoss, BatchNorm, GraphPool, Dense, GraphGather, BetaShare, LayerSplitter, SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Weights, Feature, AlphaShare

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from sluice_data import load_sluice

from sluice_tg_models import two_layer_sluice, hard_param_mt, three_layer_dense

graph_conv_model = two_layer_sluice

# Load Tox21 dataset
sluice_tasks, sluice_datasets, transformers = load_sluice()
train_dataset, valid_dataset, test_dataset = sluice_datasets
print("train dataset shape:")
print(train_dataset.get_shape)
print("valid dataset shape")
print(valid_dataset.get_shape)
print("test dataset shape")
print(test_dataset.get_shape)

# Fit models
metric = dc.metrics.Metric(dc.metrics.r2_score, np.mean, mode="classification")

# Batch size of models
batch_size = 100

model, generator, labels, task_weights = graph_conv_model(
    batch_size, sluice_tasks, minimizer=1)

print('labels')
print(labels)

model.fit_generator(
    generator(train_dataset, batch_size, epochs=100), checkpoint_interval=200)

print("Evaluating model")
train_scores = model.evaluate_generator(
    generator(train_dataset, batch_size), [metric],
    labels=labels,
    weights=[task_weights])
valid_scores = model.evaluate_generator(
    generator(valid_dataset, batch_size), [metric],
    labels=labels,
    weights=[task_weights])
test_scores = model.evaluate_generator(
    generator(test_dataset, batch_size), [metric],
    labels=labels,
    weights=[task_weights])

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

print('Test scores')
print(test_scores)
