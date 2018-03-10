import os
import pickle

import numpy as np
import ray
import tensorflow as tf
from ray.rllib.agent import Agent
from ray.rllib.asq.asq_evaluator import ASQEvaluator
from ray.rllib.optimizers import EnsembleOptimizer
from ray.tune.result import TrainingResult

DEFAULT_CONFIG = dict(
    # Whether to use dueling dqn
    dueling=True,
    # Hidden layer sizes of the state and action value networks
    hiddens=[256],
    conv_filters=[
        [16, [4, 4], 2],
        [32, [4, 4], 2],
        [512, [11, 11], 1],
    ],
    # Config options to pass to the model constructor
    model={
        "grayscale": True,
        "zero_mean": False,
        "dim": 42,
    },
    # Discount factor for the MDP
    gamma=0.99,
    # Arguments to pass to the env creator
    env_config={},
    # Number of env steps to optimize for before returning
    timesteps_per_iteration=1000,
    # # Max num timesteps for exploration schedule
    exploration_timesteps=200000,
    # # Final value of random action probability
    exploration_final_eps=0.01,
    # How many steps to sample before learning starts
    learning_starts=10000,
    # Alpha parameter for prioritized replay buffer
    prioritized_replay_alpha=0.6,
    # Initial value of beta for prioritized replay buffer
    prioritized_replay_beta0=0.4,
    # Number of timesteps for prioritized beta schedule
    prioritized_replay_timesteps=2000000,
    # Epsilon to add to the TD errors when updating priorities.
    prioritized_replay_eps=1e-6,
    # Size of the replay buffer
    buffer_size=50000,
    # If True prioritized replay buffer will be used
    prioritized_replay=True,
    # Learning rate for adam optimizer
    learning_rate=1e-4,
    # Update the replay buffer with this many samples at once
    sample_rollout_steps=4,
    # Size of a batched sampled from replay buffer for training
    sample_batch_size=32,
    # If not None, clip gradients during optimization at this value
    grad_norm_clipping=10,
    # Arguments to pass to the rllib optimizer
    optimizer={},
    # Smooth the current average reward over this many previous episodes
    smoothing_num_episodes=100,
    # Arguments to pass to tensorflow
    tf_session_args={
        "device_count": {"CPU": 1},
        "log_device_placement": False,
        "allow_soft_placement": True,
        "inter_op_parallelism_threads": 1,
        "intra_op_parallelism_threads": 1,
    },
    # Arguments to pass to tensorflow
    tf_remote_session_args={
        "device_count": {"CPU": 1, "GPU": 1},
        "log_device_placement": False,
        "allow_soft_placement": True,
        "inter_op_parallelism_threads": 1,
        "intra_op_parallelism_threads": 1,
    },
    # Number of ensemble models
    num_models=1,
)


class ASQAgent(Agent):
    _agent_name = "ASQ"
    _default_config = DEFAULT_CONFIG

    def _init(self):
        self.local_evaluator = ASQEvaluator(self.registry, self.env_creator, self.config)
        remote_cls = ray.remote(num_cpus=1, num_gpus=1)(ASQEvaluator)
        self.remote_evaluators = [
            remote_cls.remote(self.registry, self.env_creator, self.config, model_idx)
            for model_idx in range(self.config["num_models"])]
        self.optimizer = EnsembleOptimizer(self.config["optimizer"], self.local_evaluator, self.remote_evaluators)
        self.saver = tf.train.Saver(max_to_keep=None)
        self.global_timestep = 0

    def _train(self):
        start_timestep = self.global_timestep
        while self.global_timestep - start_timestep < self.config["timesteps_per_iteration"]:
            if self.global_timestep < self.config["learning_starts"]:
                self._populate_replay_buffer()
            else:
                self.optimizer.step()
            stats = self._update_global_stats()
        mean_episode_reward = 0.0
        mean_episode_length = 0.0
        num_episodes = 0
        exploration = None
        for s in stats:
            mean_episode_reward += s["mean_episode_reward"] / len(stats)
            mean_episode_length += s["mean_episode_length"] / len(stats)
            num_episodes += s["num_episodes"]
            exploration = s["exploration"]
        result = TrainingResult(
            episode_reward_mean=mean_episode_reward,
            episode_len_mean=mean_episode_length,
            episodes_total=num_episodes,
            timesteps_this_iter=self.global_timestep - start_timestep,
            info=dict({
                "exploration": exploration,
            }, **self.optimizer.stats()))
        return result

    def _update_global_stats(self):
        stats = ray.get([e.stats.remote() for e in self.remote_evaluators])
        new_timestep = sum(s["local_timestep"] for s in stats)
        assert new_timestep > self.global_timestep, new_timestep
        self.global_timestep = new_timestep
        self.local_evaluator.set_global_timestep(self.global_timestep)
        for e in self.remote_evaluators:
            e.set_global_timestep.remote(self.global_timestep)
        return stats

    def _populate_replay_buffer(self):
        for e in self.remote_evaluators:
            e.sample.remote(no_replay=True)

    def _save(self, checkpoint_dir):
        checkpoint_path = self.saver.save(
            self.local_evaluator.sess,
            os.path.join(checkpoint_dir, "checkpoint"),
            global_step=self.iteration)
        extra_data = [
            self.local_evaluator.save(),
            ray.get([e.save.remote() for e in self.remote_evaluators]),
            self.global_timestep]
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.saver.restore(self.local_evaluator.sess, checkpoint_path)
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        self.local_evaluator.restore(extra_data[0])
        ray.get([e.restore.remote(d) for (d, e) in zip(extra_data[1], self.remote_evaluators)])
        self.global_timestep = extra_data[2]

    def compute_action(self, observation):
        return self.local_evaluator.asq_graph.act(self.local_evaluator.sess, np.asarray(observation)[None], 0.0)[0]
