from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import queue
import numpy as np

import ray
from ray.rllib.bc.experience_dataset import ExperienceDataset
from ray.rllib.bc.policy import BCPolicy
from ray.rllib.dqn.common.wrappers import wrap_dqn
from ray.rllib.models import ModelCatalog
from ray.rllib.optimizers import Evaluator


class BCEvaluator(Evaluator):
    def __init__(self, registry, env_creator, config, logdir):
        env = wrap_dqn(registry, env_creator(config["env_config"]), config["model"])
        self.env = env
        self.dataset = ExperienceDataset(config["dataset_path"])
        print("env is", env)
        print("config is", config["model"])
        print("observation space", env.observation_space.shape)
        self.policy = BCPolicy(registry, env.observation_space.shape, env.action_space, config)
        self.config = config
        self.logdir = logdir
        self.metrics_queue = queue.Queue()

        self.local_timestep = 0
        self.episode_rewards = [0.0]
        self.episode_lengths = [0.0]
        self.obs = self.env.reset()

    def sample(self):
        self._step()
        return self.dataset.sample(self.config["batch_size"])

    def stats(self):
        mean_100ep_reward = round(np.mean(self.episode_rewards[-11:-1]), 5)
        mean_100ep_length = round(np.mean(self.episode_lengths[-11:-1]), 5)
        return {
            "mean_100ep_reward": mean_100ep_reward,
            "mean_100ep_length": mean_100ep_length,
            "num_episodes": len(self.episode_rewards),
            "local_timestep": self.local_timestep,
        }

    def _step(self):
        """Takes a single step, and returns the result of the step."""
        action = self.policy.compute(self.obs)[0][0]
        new_obs, rew, done, _ = self.env.step(action)
        ret = (self.obs, action, rew, new_obs, float(done))
        self.obs = new_obs
        self.episode_rewards[-1] += rew
        self.episode_lengths[-1] += 1
        if done:
            self.obs = self.env.reset()
            self.episode_rewards.append(0.0)
            self.episode_lengths.append(0.0)
        self.local_timestep += 1
        return ret

    def compute_gradients(self, samples):
        gradient, info = self.policy.compute_gradients(samples)
        self.metrics_queue.put({"num_samples": info["num_samples"], "loss": info["loss"]})
        return gradient

    def apply_gradients(self, grads):
        self.policy.apply_gradients(grads)

    def get_weights(self):
        return self.policy.get_weights()

    def set_weights(self, params):
        self.policy.set_weights(params)

    def save(self):
        weights = self.get_weights()
        return pickle.dumps({
            "weights": weights})

    def restore(self, objs):
        objs = pickle.loads(objs)
        self.set_weights(objs["weights"])

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed


RemoteBCEvaluator = ray.remote(BCEvaluator)
GPURemoteBCEvaluator = ray.remote(num_gpus=1)(BCEvaluator)
