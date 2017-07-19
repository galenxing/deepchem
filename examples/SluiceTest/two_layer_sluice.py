import numpy as np
import six


from deepchem.models.tensorgraph import TensorGraph
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Input, GraphConv, Add, SluiceLoss, BatchNorm, GraphPool, Dense, GraphGather, BetaShare,  LayerSplitter, SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Weights, Feature, AlphaShare


np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)

model_dir = "/tmp/graph_conv"


def to_one_hot(y, n_classes=20):
    n_samples = np.shape(y)[0]
    y_hot = np.zeros((n_samples, n_classes), dtype='float64')
    for row, value in enumerate(y):
        y_hot[row, value] = 1
    return y_hot


def graph_conv_model(batch_size, tasks):
    model = TensorGraph(model_dir=model_dir,
                        batch_size=batch_size, use_queue=False)
    sluice_cost = []

    X1 = Feature(shape=(None, 1))
    X2 = Feature(shape=(None, 1))

    d1a = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X1])
    d1b = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X2])

    d1c = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1a])
    d1d = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1b])

    sluice_cost.append(d1c)
    sluice_cost.append(d1d)

    as1 = AlphaShare(in_layers=[d1c, d1d])
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

    b1 = BetaShare(in_layers=[ls1a, ls2a])
    b2 = BetaShare(in_layers=[ls1b, ls2b])

    count = 0
    costs = []
    labels = []

    for task in tasks:
        if count < len(tasks) / 2:
            classification = Dense(
                out_channels=20, activation_fn=None, in_layers=[b1])
            label = Label(shape=(None, 20))
        else:
            classification = Dense(
                out_channels=20, activation_fn=None, in_layers=[b2])
            label = Label(shape=(None, 20))
        count += 1
        softmax = SoftMax(in_layers=[classification])
        model.add_output(softmax)

        labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)

    s_cost = SluiceLoss(in_layers=sluice_cost)
    entropy = Concat(in_layers=costs)
    task_weights = Weights(shape=(None, len(tasks)))
    loss = WeightedError(in_layers=[entropy, task_weights])
    loss = Add(in_layers=[loss, s_cost])
    model.set_alphas([as1, as2])
    model.set_betas([b1, b2])

    model.set_loss(loss)

    def feed_dict_generator(dataset, batch_size, epochs=1):
        for epoch in range(epochs):
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                    dataset.iterbatches(batch_size, pad_batches=True)):
                d = {}
                d[X1] = X_b
                d[X2] = X_b
                for index, label in enumerate(labels):
                    d[label] = to_one_hot(y_b[:, index])
                d[task_weights] = w_b
                yield d
    return model, feed_dict_generator, labels, task_weights
