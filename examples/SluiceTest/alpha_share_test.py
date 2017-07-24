import tensorflow as tf
from pprint import pprint
import numpy as np


def create_tensor(in_layers=None, set_tensors=True, **kwargs):
    sess = tf.Session()

    inputs = []

    #tower_1 = tf.reshape(tf.range(40), (4, 10))
    #tower_2 = tf.reshape(tf.range(40, 80), (4, 10))

    tower_1 = tf.reshape(tf.range(8), (4,2))
    tower_2 = tf.reshape(tf.range(8, 16), (4,2))
    inputs.append(tower_1)
    inputs.append(tower_2)

    # create subspaces
    subspaces = []

    pprint(sess.run(inputs))

    original_cols = int(inputs[0].get_shape()[-1].value)
    pprint("orignal_cols")
    pprint(original_cols)
    subspace_size = int(original_cols / 2)
    pprint("subspace_size")
    pprint(subspace_size)

    for input_tensor in inputs:
        subspaces.append(tf.reshape(input_tensor[:, :subspace_size], [-1]))
        subspaces.append(tf.reshape(input_tensor[:, subspace_size:], [-1]))
    n_alphas = len(inputs) * 2
    subspaces = tf.reshape(tf.stack(subspaces), [n_alphas, -1])

    pprint("subspaces")
    pprint(sess.run(subspaces))

    count = 0
    out_tensor = []
    tmp_tensor = []
    print("next:")
    for row in range(n_alphas):
        tmp_tensor.append(tf.reshape(subspaces[row,], [-1, subspace_size]))
        count += 1
        if(count ==2):
            out_tensor.append(tf.concat(tmp_tensor,1))
            tmp_tensor =[]
            count =0
    
    out_tensor = tf.stack(out_tensor)

    pprint("out tensor")
    pprint(sess.run(out_tensor))

create_tensor()
