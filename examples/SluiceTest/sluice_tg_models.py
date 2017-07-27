import numpy as np
import six


from deepchem.models.tensorgraph import TensorGraph
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Input, GraphConv, Add, L2Loss, Constant, Multiply,  SluiceLoss, BatchNorm, GraphPool, Dense, GraphGather, BetaShare,  LayerSplitter, SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Weights, Feature, AlphaShare
import time

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
current_milli_time = lambda: "_" + str(int(round(time.time() * 1000)))


def to_one_hot(y, n_classes=5):
    n_samples = np.shape(y)[0]
    y_hot = np.zeros((n_samples, n_classes), dtype='float64')
    for row, value in enumerate(y):
        y_hot[row, value] = 1
    return y_hot

def three_layer_dense(batch_size, tasks, mode = "classification"):
    model = TensorGraph(model_dir='graphconv/three_layer_dense'+str(current_milli_time()),
                        batch_size=batch_size, use_queue=False, tensorboard=True)
    sluice_cost = []

    X1 = Feature(shape=(None, 1))
    X2 = Feature(shape=(None, 1))

    d1a = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X1])
    d1b = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X2])

    d1c = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1a])
    d1d = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1b])

    d2a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1c])
    d2b = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1d])

    d3a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d2a])
    d3b = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d2b])

    count = 0
    costs = []
    labels = []

    if mode == "regression":
        for task in tasks:
            if count < len(tasks) / 2:
                regression = Dense(
                    out_channels=1, activation_fn=None, in_layers=[d3a])
            else:
                regression = Dense(
                    out_channels=1, activation_fn=None, in_layers=[d3b])

            model.add_output(regression)
            count += 1

            label = Label(shape= (None, 1))
            labels.append(label)
            cost = L2Loss(in_layers= [label,regression])
            costs.append(cost)

    elif mode == "classification":
        for task in tasks:
            if count < len(tasks) / 2:
                classification = Dense(
                    out_channels=5, activation_fn=None, in_layers=[d3a])
            else:
                classification = Dense(
                    out_channels=5, activation_fn=None, in_layers=[d3b])

            count += 1
            softmax = SoftMax(in_layers = [classification])
            model.add_output(softmax)

            label = Label(shape= (None, 5))
            labels.append(label)

            cost = SoftMaxCrossEntropy(in_layers= [label,classification])
            costs.append(cost)

    entropy = Concat(in_layers=costs, axis = -1)
    task_weights = Weights(shape=(None, len(tasks)))
    loss = WeightedError(in_layers=[entropy, task_weights])
    model.set_loss(loss)

    def feed_dict_generator(dataset, batch_size, epochs=1):
        for epoch in range(epochs):
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                    dataset.iterbatches(batch_size, pad_batches=True)):
                d = {}
                d[X1] = X_b
                d[X2] = X_b
                for index, label in enumerate(labels):
                    if mode == "classification":
                        d[label] = to_one_hot(y_b[:, index])
                    elif mode == "regression":
                        d[label] = np.expand_dims(y_b[:, index], -1)
                d[task_weights] = w_b
                yield d
    return model, feed_dict_generator, labels, task_weights

def hard_param_mt_regression(batch_size, tasks):
    model = TensorGraph(model_dir='graphconv/hard_param_mt' + str(current_milli_time()),
                        batch_size=batch_size, use_queue=False, tensorboard=True)
    sluice_cost = []

    X1 = Feature(shape=(None, 1))

    d1a = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X1])

    d1c = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1a])

    d2a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1c])

    d3a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d2a])

    costs = []
    labels = []
    count=0
    for task in tasks:
        if count < len(tasks) / 2:
            regression = Dense(
                out_channels=1, activation_fn=None, in_layers=[d3a])
        else:
            regression = Dense(
                out_channels=1, activation_fn=None, in_layers=[d3a])

        model.add_output(regression)
        count += 1

        label = Label(shape= (None, 1))
        labels.append(label)
        cost = L2Loss(in_layers= [label,regression])
        costs.append(cost)

    entropy = Concat(in_layers=costs, axis = -1)
    task_weights = Weights(shape=(None, len(tasks)))
    loss = WeightedError(in_layers=[entropy, task_weights])
    model.set_loss(loss)

    def feed_dict_generator(dataset, batch_size, epochs=1):
        for epoch in range(epochs):
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                    dataset.iterbatches(batch_size, pad_batches=True)):
                d = {}
                d[X1] = X_b
                for index, label in enumerate(labels):
                    d[label] = np.expand_dims(y_b[:, index],-1)
                d[task_weights] = w_b
                yield d
    return model, feed_dict_generator, labels, task_weights


def three_layer_sluice_regression(batch_size, tasks, minimizer=1):
    model = TensorGraph(model_dir='graphconv/three_layer_sluice' + str(minimizer)+str(current_milli_time()),
                        batch_size=batch_size, use_queue=False, tensorboard=True)
    sluice_cost = []

    X1 = Feature(shape=(None, 1))
    X2 = Feature(shape=(None, 1))

    d1a = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X1])
    d1b = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X2])

    sluice_cost.append(d1a)
    sluice_cost.append(d1b)

    as1 = AlphaShare(in_layers=[d1a, d1b])
    ls1a = LayerSplitter(in_layers=[as1], tower_num=0)
    ls1b = LayerSplitter(in_layers=[as1], tower_num=1)

    d2a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls1a])
    d2b = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls1b])

    sluice_cost.append(d2a)
    sluice_cost.append(d2b)
    as2 = AlphaShare(in_layers=[d2a, d2b])
    ls2a = LayerSplitter(in_layers=[as2], tower_num=0)
    ls2b = LayerSplitter(in_layers=[as2], tower_num=1)

    d3a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls2a])
    d3b = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls2b])

    d4a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d3a])
    d4b = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d3b])

    sluice_cost.append(d4a)
    sluice_cost.append(d4b)
    as3 = AlphaShare(in_layers=[d4a, d4b])
    ls3a = LayerSplitter(in_layers=[as3], tower_num=0)
    ls3b = LayerSplitter(in_layers=[as3], tower_num=1)

    count = 0
    costs = []
    labels = []

    for task in tasks:
        if count < len(tasks) / 2:
            regression = Dense(
                out_channels=1, activation_fn=None, in_layers=[ls3a])
        else:
            regression = Dense(
                out_channels=1, activation_fn=None, in_layers=[ls3b])

        model.add_output(regression)
        count += 1

        label = Label(shape= (None, 1))
        labels.append(label)
        cost = L2Loss(in_layers= [label,regression])
        costs.append(cost)

    s_cost = SluiceLoss(in_layers=sluice_cost)
    entropy = Concat(in_layers=costs, axis = -1)
    task_weights = Weights(shape=(None, len(tasks)))
    total_loss = WeightedError(in_layers=[entropy, task_weights])

    minimizer = Constant(minimizer)
    s_cost = Multiply(in_layers=[minimizer, s_cost])

    new_loss = Add(in_layers=[total_loss, s_cost])

    model.set_loss(new_loss)

    def feed_dict_generator(dataset, batch_size, epochs=1):
        for epoch in range(epochs):
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                    dataset.iterbatches(batch_size, pad_batches=True)):
                d = {}
                d[X1] = X_b
                d[X2] = X_b
                for index, label in enumerate(labels):
                    d[label] = np.expand_dims(y_b[:, index], -1)
                d[task_weights] = w_b
                yield d
    return model, feed_dict_generator, labels, task_weights
