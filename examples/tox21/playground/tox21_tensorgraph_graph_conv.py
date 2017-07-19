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
    SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Weights, Feature

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from datasets import load_tox21
from record_info import record_info

model_dir = "/tmp/graph_conv"


def graph_conv_model(batch_size, tasks):
    model = TensorGraph(
        model_dir=model_dir, batch_size=batch_size, use_queue=False)
    atom_features = Feature(shape=(None, 75))
    degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
    membership = Feature(shape=(None,), dtype=tf.int32)

    deg_adjs = []
    for i in range(0, 10 + 1):
        deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
        deg_adjs.append(deg_adj)
    gc1 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[atom_features, degree_slice, membership] + deg_adjs)
    batch_norm1 = BatchNorm(in_layers=[gc1])
    gp1 = GraphPool(
        in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)
    gc2 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[gp1, degree_slice, membership] + deg_adjs)
    batch_norm2 = BatchNorm(in_layers=[gc2])

    gp2 = GraphPool(
        in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)
    dense = Dense(out_channels=128, activation_fn=None, in_layers=[gp2])
    batch_norm3 = BatchNorm(in_layers=[dense])
    gg1 = GraphGather(
        batch_size=batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)

    costs = []
    labels = []
    for task in tasks:
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[gg1])

        softmax = SoftMax(in_layers=[classification])
        model.add_output(softmax)

        label = Label(shape=(None, 2))
        labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)

    entropy = Concat(in_layers=costs)
    task_weights = Weights(shape=(None, len(tasks)))
    loss = WeightedError(in_layers=[entropy, task_weights])
    model.set_loss(loss)

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


def run_tensorgraph_graph_conv(tasks, input_file_name, task_num=0, epochs=10, split='random', trials=1, output_file_name='temp.csv'):
    # Load Tox21 dataset
    tox21_tasks, tox21_datasets, transformers = load_tox21(
        input_file_name=input_file_name, tasks=tasks, featurizer='GraphConv', split=split)
    train_dataset, valid_dataset, test_dataset = tox21_datasets
    print(train_dataset.data_dir)
    print(valid_dataset.data_dir)
    print(test_dataset.data_dir)
    for x in range(0, trials):
        metric = dc.metrics.Metric(
            dc.metrics.roc_auc_score, np.mean, mode="classification")
        # Batch size of mode
        batch_size = 100
        model, generator, labels, task_weights = graph_conv_model(batch_size,
                                                                  tox21_tasks)
        model.fit_generator(
            generator(train_dataset, batch_size, epochs=epochs))
        print("Evaluating model")
        train_scores = model.evaluate_generator(
            generator(train_dataset, batch_size), [metric],
            transformers,
            labels,
            # weights=[task_weights], per_task_metrics = True)
            weights=[task_weights])
        valid_scores = model.evaluate_generator(
            generator(valid_dataset, batch_size), [metric],
            transformers,
            labels,
            weights=[task_weights])
      # weights=[task_weights], per_task_metrics = True)
        test_scores, all_task_scores = model.evaluate_generator(
            generator(test_dataset, batch_size), [metric],
            transformers,
            labels,
            weights=[task_weights], per_task_metrics=True)

        print("Train scores")
        print(train_scores)

        print("Validation scores")
        print(valid_scores)

        print("Test scores")
        print(test_scores)

        all_task_scores = next(iter(all_task_scores.values()))[task_num]
        record_info(output_file_name, train_scores, valid_scores,
                    all_task_scores, epochs, len(tasks), tasks)


run_tensorgraph_graph_conv(tasks=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase'], input_file_name="../../../datasets/tox21.csv.gz")


# test to see if they are all positive or negative
# valid_y = valid_dataset.y
# print("\n\n\n\nValid y shapee")
# print(valid_y.shape)
# for y in valid_y:
# if ((y==y[0]).all()) != True:
#   print(y[0])

# train_y = train_dataset.y
# for y in train_y:
# if ((y==y[0]).all()) != True:
# print(y[0])

# test_y = test_dataset.y
# for y in test_y:
#  if ((y==y[0]).all()) != True:
#    print(y[0])

# Fit models
