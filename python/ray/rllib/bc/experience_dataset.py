from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import json
import itertools
import pickle

import numpy as np


class ExperienceDataset(object):
    def __init__(self, dataset_path):
        """Create dataset of experience to imitate.

        Parameters
        ----------
        dataset_path:
          Path of file containing the database as pickled list of trajectories,
          each trajectory being a list of steps,
          each step containing the observation and action as its first two elements.
          The file must be available on each machine used by a BCEvaluator.
        """
#        self._dataset = list(itertools.chain.from_iterable(pickle.load(open(dataset_path, "rb"))))
        self._dataset = []
        for path, sample_frac in dataset_path.items():
            for line in open(path).read().split("\n"):
                line = line.strip()
                if line and random.random() < sample_frac:
                    self._dataset.append(json.loads(line))
            print("Loaded dataset size", len(self._dataset))

    def sample(self, batch_size):
        indexes = np.random.choice(len(self._dataset), batch_size)
        samples = {
            'observations': [self._dataset[i]["obs"] for i in indexes],
            'actions': [self._dataset[i]["action"] for i in indexes],
        }
        return samples
