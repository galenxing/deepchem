"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc

# should be able to delete next line
from deepchem.splits import splitters


def load_tox21(tasks, input_file_name, featurizer='ECFP', split='index'):
    """Load Tox21 datasets. Does not do train/test split"""
    # Featurize Tox21 dataset
    print("About to featurize Tox21 dataset.")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_file = os.path.join(
        current_dir, input_file_name)
    # tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
    #               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
    #               'SR-HSE', 'SR-MMP', 'SR-p53','HIV_active']
    tox21_tasks = tasks
    if featurizer == 'ECFP':
        featurizer_func = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer_func = dc.feat.ConvMolFeaturizer()
    loader = dc.data.CSVLoader(
        tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer_func)
    dataset = loader.featurize(dataset_file, shard_size=8192)
    dataset_shape = dataset.get_shape()
    print("the shape of the dataset!")
    print(dataset_shape)

    # Initialize transformers
    transformers = [
        dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

    print("About to transform data")
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    splitters = {'index': dc.splits.IndexSplitter(),
                 'random': dc.splits.RandomSplitter(),
                 'scaffold': dc.splits.ScaffoldSplitter(),
                 'butina': dc.splits.ButinaSplitter(),
                 'fingerprint': dc.splits.FingerprintSplitter()}
    splitter = splitters[split]

    #train_dir = "/tmp/" + str(len(tasks)) + '_train_rand'
    #valid_dir = "/tmp/" + str(len(tasks)) + '_valid_rand'
    #test_dir = "/tmp/" + str(len(tasks)) + '_test_rand'

    #train, valid, test = splitter.train_valid_test_split(
    #    dataset, train_dir=train_dir, valid_dir=valid_dir, test_dir=test_dir)

    train = dc.data.DiskDataset(data_dir='/tmp/4_train_rand')
    valid = dc.data.DiskDataset(data_dir='/tmp/4_valid_rand')
    test = dc.data.DiskDataset(data_dir='/tmp/4_test_rand')

    print(train.shape)
    return tox21_tasks, (train, valid, test), transformers
