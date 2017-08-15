"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from itertools import permutations
np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from tox21_datasets import load_tox21
from gc_sluice_network import graph_conv_sluice
from record_info import record_info

tox21_tasks, tox21_datasets, transformers = load_tox21(
    featurizer='GraphConv', split='fingerprint')
train_dataset, valid_dataset, test_dataset = tox21_datasets

print(train_dataset.data_dir)
print(valid_dataset.data_dir)


class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

batch_size = 50
sluice_combos = list(perm_unique([1,1,1,0,0,0,0,0]))
weights = [0, 0.25, 1, 2]
epoch = 10
for weight in weights:
	for combo in sluice_combos:
		for x in range(0,3):
			print(combo)
			print(weight)
			# Load Tox21 dataset
			# Fit models
			metric = dc.metrics.Metric(
			    dc.metrics.roc_auc_score, np.mean, mode="classification")
			# Batch size of models

			model, generator, labels, task_weights = graph_conv_sluice(
			    n_tasks=len(tox21_tasks),
			    batch_size=batch_size,
			    mode='classification',
			    minimizer=weight,
			    sluice_layout=combo,
			    tensorboard=True)
			model.fit_generator(generator(train_dataset, batch_size, epochs=epoch))

			print("Evaluating model")
			train_scores = model.evaluate_generator(
			    generator(train_dataset, batch_size), [metric],
			    transformers,
			    labels,
			    weights=[task_weights],
			    per_task_metrics = True)

			valid_scores = model.evaluate_generator(
			    generator(valid_dataset, batch_size), [metric],
			    transformers,
			    labels,
			    weights=[task_weights],
			    per_task_metrics = True)

			print("Train scores")
			print(train_scores)

			print("Validation scores")
			print(valid_scores)
			record_info(file_name= 'sluice_hp_search_without_epochs.csv', train= train_scores, valid = valid_scores, weight = weight, combo= combo)

