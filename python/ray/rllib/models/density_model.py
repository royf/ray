import numpy as np
from ray.rllib.models.fast_cts import CTSDensityModel


class DensityModel(object):
    def __init__(self, num_actions=None):
        self.model_actions = num_actions is not None
        if self.model_actions:
            self.action_count = np.zeros(num_actions)
            self.models = [CTSDensityModel() for _ in range(num_actions)]
        else:
            self.model = CTSDensityModel()

    def get_pseudocount(self, observations):
        pseudocounts = [None] * len(observations)
        for i in range(len(observations)):
            observation = observations[i]
            if self.model_actions:
                log_total_count = np.log(self.action_count.sum())
                pseudocounts[i] = [
                    np.log(self.action_count[action]) - log_total_count
                    + self.models[action].get_pseudocount(observation)
                    for action in range(len(self.action_count))]
            else:
                pseudocounts[i] = self.model.get_pseudocount(observation)
            # recoding_prob = np.exp(log_recoding_prob)
            # prob_ratio = np.exp(log_recoding_prob - log_prob)
            # pseudocounts[i] = (1. - recoding_prob) / max(prob_ratio - 1., 1e-10)
        return pseudocounts

    def update(self, observations, actions=None):
        for i in range(len(observations)):
            observation = observations[i]
            if self.model_actions:
                action = actions[i]
                self.action_count[action] += 1
                self.models[action].update(observation)
            else:
                self.model.update(observation)
