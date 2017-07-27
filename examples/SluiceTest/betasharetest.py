import numpy as np
import tensorflow as tf

from pprint import pprint


def test():
    sess = tf.Session()
    inputs = np.arange(24).reshape(3,4,2)

    pprint(inputs)
    subspaces = []

    original_cols = len(inputs[0][0])
    print(original_cols)

    for input_tensor in inputs:
        subspaces.append(tf.reshape(input_tensor, [-1]))
    n_betas = len(inputs)
    subspaces = tf.reshape(tf.stack(subspaces), [n_betas, -1])

    pprint(sess.run(subspaces))
   # exit()

    betas = tf.Variable(tf.random_normal([1, n_betas]), name='betas')
    #out_tensor = tf.matmul(betas, subspaces)
   # self.betas = betas
    out_tensor = tf.reshape(subspaces, [-1, original_cols])
    pprint(sess.run(out_tensor))
    return out_tensor

test()