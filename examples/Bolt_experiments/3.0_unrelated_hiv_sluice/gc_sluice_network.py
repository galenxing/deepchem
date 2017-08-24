import numpy as np
import six
import tensorflow as tf

from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveLayer, WeaveGather, \
    Combine_AP, Separate_AP, DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, DAGGather, DTNNExtract
from deepchem.models.tensorgraph.layers import Dense, Concat, SoftMax, SoftMaxCrossEntropy, GraphConv, BatchNorm, \
    GraphPool, GraphGather, WeightedError, BatchNormalization, AlphaShare, LayerSplitter, SluiceLoss, Add, Multiply, Constant
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator

model_dir = 'tmp/graphconv'


def graph_conv_sluice(batch_size, n_tasks, sluice_layout, minimizer, mode,
                      tensorboard):
  model = TensorGraph(
      model_dir=model_dir,
      batch_size=batch_size,
      use_queue=False,
      tensorboard=tensorboard)
  """
  Building graph structures:
  """
  as_started = False
  s_layout = sluice_layout
  sluice_cost = []

  atom_features = Feature(shape=(None, 75))
  degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
  membership = Feature(shape=(None,), dtype=tf.int32)

  deg_adjs = []
  for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)
  #######################################################################
  gc1 = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[atom_features, degree_slice, membership] + deg_adjs)

  prev_node = gc1
  prev_layer = None
  #######################################################################
  if s_layout[0] == 1:
    sluice_cost.append(gc1)
    as1 = AlphaShare(in_layers=[gc1, gc1])
    ls1a = LayerSplitter(in_layers=as1, tower_num=0)
    ls1b = LayerSplitter(in_layers=as1, tower_num=1)
    as_started = True
    prev_layer = [ls1a, ls1b]
  #######################################################################
  if as_started:
    batch_norm1a = BatchNorm(in_layers=prev_layer[0])
    batch_norm1b = BatchNorm(in_layers=prev_layer[1])
    prev_layer = [batch_norm1a, batch_norm1b]
  else:
    batch_norm1a = BatchNorm(in_layers=prev_node)
    prev_node = batch_norm1a
  #######################################################################
  if s_layout[1] == 1:
    if not as_started:
      prev_layer = [prev_node, prev_node]
    sluice_cost.append(prev_layer[0])
    sluice_cost.append(prev_layer[1])
    as2 = AlphaShare(in_layers=prev_layer)
    ls2a = LayerSplitter(in_layers=as2, tower_num=0)
    ls2b = LayerSplitter(in_layers=as2, tower_num=1)
    prev_layer = [ls2a, ls2b]
    as_started = True
  #######################################################################
  if as_started:
    gp1a = GraphPool(
        in_layers=[prev_layer[0], degree_slice, membership] + deg_adjs)
    gp1b = GraphPool(
        in_layers=[prev_layer[1], degree_slice, membership] + deg_adjs)
    prev_layer = [gp1a, gp1b]
  else:
    gp1a = GraphPool(in_layers=[prev_node, degree_slice, membership] + deg_adjs)
    prev_node = gp1a
  #######################################################################
  if s_layout[2] == 1:
    if not as_started:
      prev_layer = [prev_node, prev_node]
    sluice_cost.append(prev_layer[0])
    sluice_cost.append(prev_layer[1])
    as3 = AlphaShare(in_layers=prev_layer)
    ls3a = LayerSplitter(in_layers=as3, tower_num=0)
    ls3b = LayerSplitter(in_layers=as3, tower_num=1)
    prev_layer = [ls3a, ls3b]
    as_started = True
  #######################################################################
  if as_started:
    gc2a = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[prev_layer[0], degree_slice, membership] + deg_adjs)
    gc2b = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[prev_layer[1], degree_slice, membership] + deg_adjs)
    prev_layer = [gc2a, gc2b]
  else:
    gc2a = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[prev_node, degree_slice, membership] + deg_adjs)
    prev_node = gc2a
  #######################################################################
  if s_layout[3] == 1:
    if not as_started:
      prev_layer = [prev_node, prev_node]
    sluice_cost.append(prev_layer[0])
    sluice_cost.append(prev_layer[1])
    as4 = AlphaShare(in_layers=prev_layer)
    ls4a = LayerSplitter(in_layers=as4, tower_num=0)
    ls4b = LayerSplitter(in_layers=as4, tower_num=1)
    prev_layer = [ls4a, ls4b]
    as_started = True
  #######################################################################
  if as_started:
    batch_norm2a = BatchNorm(in_layers=[prev_layer[0]])
    batch_norm2b = BatchNorm(in_layers=[prev_layer[1]])
    prev_layer = [batch_norm2a, batch_norm2b]
  else:
    batch_norm2a = BatchNorm(in_layers=[prev_node])
    prev_node = batch_norm2a
  #######################################################################
  if s_layout[4] == 1:
    if not as_started:
      prev_layer = [prev_node, prev_node]
    sluice_cost.append(prev_layer[0])
    sluice_cost.append(prev_layer[1])
    as5 = AlphaShare(in_layers=prev_layer)
    ls5a = LayerSplitter(in_layers=as5, tower_num=0)
    ls5b = LayerSplitter(in_layers=as5, tower_num=1)
    prev_layer = [ls5a, ls5b]
    as_started = True
  #######################################################################
  if as_started:
    gp2a = GraphPool(
        in_layers=[prev_layer[0], degree_slice, membership] + deg_adjs)
    gp2b = GraphPool(
        in_layers=[prev_layer[1], degree_slice, membership] + deg_adjs)
    prev_layer = [gp2a, gp2b]
  else:
    gp2a = GraphPool(in_layers=[prev_node, degree_slice, membership] + deg_adjs)
    prev_node = gp2a
  #######################################################################
  if s_layout[5] == 1:
    if not as_started:
      prev_layer = [prev_node, prev_node]
    sluice_cost.append(prev_layer[0])
    sluice_cost.append(prev_layer[1])
    as6 = AlphaShare(in_layers=prev_layer)
    ls6a = LayerSplitter(in_layers=as6, tower_num=0)
    ls6b = LayerSplitter(in_layers=as6, tower_num=1)
    prev_layer = [ls6a, ls6b]
    as_started = True
  #######################################################################
  if as_started:
    densea = Dense(
        out_channels=128, activation_fn=None, in_layers=[prev_layer[0]])
    denseb = Dense(
        out_channels=128, activation_fn=None, in_layers=[prev_layer[1]])
    prev_layer = [densea, denseb]
  else:
    densea = Dense(out_channels=128, activation_fn=None, in_layers=[prev_node])
    prev_node = densea
  #######################################################################
  if s_layout[6] == 1:
    if not as_started:
      prev_layer = [prev_node, prev_node]
    sluice_cost.append(prev_layer[0])
    sluice_cost.append(prev_layer[1])
    as7 = AlphaShare(in_layers=prev_layer)
    ls7a = LayerSplitter(in_layers=as7, tower_num=0)
    ls7b = LayerSplitter(in_layers=as7, tower_num=1)
    prev_layer = [ls7a, ls7b]
    as_started = True
  #######################################################################
  if as_started:
    batch_norm3a = BatchNorm(in_layers=[prev_layer[0]])
    batch_norm3b = BatchNorm(in_layers=[prev_layer[1]])
    prev_layer = [batch_norm3a, batch_norm3b]
  else:
    batch_norm3a = BatchNorm(in_layers=[prev_node])
    prev_node = batch_norm3a
  #######################################################################
  if s_layout[7] == 1:
    if not as_started:
      prev_layer = [prev_node, prev_node]
    sluice_cost.append(prev_layer[0])
    sluice_cost.append(prev_layer[1])
    as8 = AlphaShare(in_layers=prev_layer)
    ls8a = LayerSplitter(in_layers=as8, tower_num=0)
    ls8b = LayerSplitter(in_layers=as8, tower_num=1)
    prev_layer = [ls8a, ls8b]
    as_started = True
  #######################################################################
  if as_started:
    gg1a = GraphGather(
        batch_size=batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[prev_layer[0], degree_slice, membership] + deg_adjs)
    gg1b = GraphGather(
        batch_size=batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[prev_layer[1], degree_slice, membership] + deg_adjs)
    prev_layer = [gg1a, gg1b]
  else:
    gg1a = GraphGather(
        batch_size=batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[prev_node, degree_slice, membership] + deg_adjs)
    prev_node = gg1a
  #######################################################################
  costs = []
  my_labels = []
  count = 0
  for task in range(n_tasks):
    if mode == 'classification':
      if count < n_tasks-1:
        if as_started:
          classification = Dense(
              out_channels=2, activation_fn=None, in_layers=[prev_layer[0]])
        else:
          classification = Dense(
              out_channels=2, activation_fn=None, in_layers=[prev_node])

        count += 1
        softmax = SoftMax(in_layers=[classification])
      else:
        if as_started:
          classification = Dense(
              out_channels=2, activation_fn=None, in_layers=[prev_layer[1]])
        else:
          classification = Dense(
              out_channels=2, activation_fn=None, in_layers=[prev_node])

        count += 1
        softmax = SoftMax(in_layers=[classification])
        minimize = Constat(HIV_minimizer)
        softmax = Multiply(in_layers= [HIV_minimizer, minimize])        

      model.add_output(softmax)

      label = Label(shape=(None, 2))
      my_labels.append(label)
      cost = SoftMaxCrossEntropy(in_layers=[label, classification])
      costs.append(cost)
    if mode == 'regression':
      regression = Dense(out_channels=1, activation_fn=None, in_layers=[gg1])
      add_output(regression)

      label = Label(shape=(None, 1))
      my_labels.append(label)
      cost = L2Loss(in_layers=[label, regression])
      costs.append(cost)
      
  s_cost = SluiceLoss(in_layers=sluice_cost)
  entropy = Concat(in_layers=costs, axis=-1)
  my_task_weights = Weights(shape=(None, n_tasks))
  loss = WeightedError(in_layers=[entropy, my_task_weights])

  minimizer = Constant(minimizer)
  s_cost = Multiply(in_layers=[minimizer, s_cost])
  new_loss = Add(in_layers=[loss, s_cost])

  model.set_loss(new_loss)

  def default_generator(dataset,
                        batch_size,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(batch_size, pad_batches=True,
                              deterministic=True)):
        d = {}
        for index, label in enumerate(my_labels):
          if mode == 'classification':
            d[label] = to_one_hot(y_b[:, index])
          if mode == 'regression':
            d[label] = np.expand_dims(y_b[:, index], -1)
        d[my_task_weights] = w_b
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        d[atom_features] = multiConvMol.get_atom_features()
        d[degree_slice] = multiConvMol.deg_slice
        d[membership] = multiConvMol.membership
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
        yield d

  return model, default_generator, my_labels, my_task_weights
