import numpy as np
from ray.rllib.models.fast_cts import CTSDensityModel


class DensityModel(object):
    def __init__(self, num_actions=None):
        self.model_actions = num_actions is not None
        if self.model_actions:
            self.action_count = np.full(num_actions, 1. / num_actions)
            self.models = [CTSDensityModel() for _ in range(num_actions)]
        else:
            self.model = CTSDensityModel()

    def get_pseudocount(self, observations):
        pseudocount = [None] * len(observations)
        for i in range(len(observations)):
            observation = observations[i]
            if self.model_actions:
                log_total_count = np.log(self.action_count.sum())
                log_total_count_p1 = np.log(self.action_count.sum() + 1)
                pseudocount[i] = [None] * len(self.action_count)
                for action in range(len(self.action_count)):
                    log_prob = self.models[action].get_log_prob(observation) + np.log(self.action_count[action]) - log_total_count
                    log_recoding_prob = self.models[action].get_log_recoding_prob(observation) + np.log(self.action_count[action] + 1) - log_total_count_p1
                    recoding_prob = np.exp(log_recoding_prob)
                    prob_ratio = min(np.exp(max(log_recoding_prob - log_prob, 1e-10)), 1e10)
                    pseudocount[i][action] = (1. - recoding_prob) / (prob_ratio - 1.)
            else:
                log_prob = self.model.get_log_prob(observation)
                log_recoding_prob = self.model.get_log_recoding_prob(observation)
                recoding_prob = np.exp(log_recoding_prob)
                prob_ratio = min(np.exp(max(log_recoding_prob - log_prob, 1e-10)), 1e10)
                pseudocount[i] = (1. - recoding_prob) / (prob_ratio - 1.)
        return pseudocount

    def update(self, observations, actions=None):
        log_prob = [None] * len(observations)
        for i in range(len(observations)):
            observation = observations[i]
            if self.model_actions:
                action = actions[i]
                self.action_count[action] += 1
                log_prob[i] = self.models[action].update(observation)
            else:
                log_prob[i] = self.model.update(observation)
        return log_prob
