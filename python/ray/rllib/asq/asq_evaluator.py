from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete
from ray.rllib.asq import models
from ray.rllib.dqn.common.schedules import LinearSchedule
from ray.rllib.dqn.common.wrappers import wrap_dqn
from ray.rllib.optimizers.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from ray.rllib.optimizers import PolicyEvaluator, SampleBatch
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.compression import pack


class ASQEvaluator(PolicyEvaluator):
    def __init__(self, registry, env_creator, config, model_idx=None):
        self.config = config
        self.model_idx = model_idx
        self.env = env_creator(self.config["env_config"])
        self.env = wrap_dqn(registry, self.env, self.config["model"], self.config["random_starts"])
        if not isinstance(self.env.action_space, Discrete):
            raise UnsupportedSpaceException("Action space {} is not supported for ASQ.".format(self.env.action_space))
        if model_idx is None:
            tf_config = tf.ConfigProto(**config["tf_session_args"])
        else:
            tf_config = tf.ConfigProto(**config["tf_remote_session_args"])
        self.sess = tf.Session(config=tf_config)
        self.asq_graph = models.ASQGraph(self.env, config, config["num_models"], model_idx is not None)
        self.sess.run(tf.global_variables_initializer())
        self.global_timestep = 0
        if model_idx is not None:
            self.exploration = LinearSchedule(
                schedule_timesteps=config["exploration_timesteps"],
                initial_p=1.0,
                final_p=config["exploration_final_eps"])
            if config["prioritized_replay"]:
                self.replay_buffer = PrioritizedReplayBuffer(
                    config["buffer_size"],
                    alpha=config["prioritized_replay_alpha"],
                    clip_rewards=config["clip_rewards"])
                self.prioritized_replay_beta_schedule = LinearSchedule(
                    config["prioritized_replay_timesteps"],
                    initial_p=config["prioritized_replay_beta0"],
                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(config["buffer_size"])
                self.prioritized_replay_beta_schedule = None
            self.local_timestep = 0
            self.episode_rewards = [0.0]
            self.episode_lengths = [0.0]
            self.saved_mean_reward = None
            self.obs = self.env.reset()

    def sample(self, no_replay=False):
        obs, actions, rewards, new_obs, dones = [], [], [], [], []
        for _ in range(self.config["sample_rollout_steps"]):
            ob, act, rew, ob1, done = self._step()
            obs.append(ob)
            actions.append(act)
            rewards.append(rew)
            new_obs.append(ob1)
            dones.append(done)
        batch = SampleBatch({"obs": [pack(np.array(o)) for o in obs], "actions": actions, "rewards": rewards,
                             "new_obs": [pack(np.array(o)) for o in new_obs], "dones": dones, "weights": np.ones_like(rewards)})
        for row in batch.rows():
            self.replay_buffer.add(row["obs"], row["actions"], row["rewards"], row["new_obs"], row["dones"], None)
        if no_replay:
            return
        if self.config["prioritized_replay"]:
            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_indexes) = self.replay_buffer.sample(
                self.config["sample_batch_size"],
                beta=self.prioritized_replay_beta_schedule.value(self.global_timestep))
            batch = SampleBatch({"obs": obses_t, "actions": actions, "rewards": rewards,
                                 "new_obs": obses_tp1, "dones": dones, "weights": weights, "batch_indexes": batch_indexes})
        else:
            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.config["sample_batch_size"])
            batch = SampleBatch({"obs": obses_t, "actions": actions, "rewards": rewards,
                                 "new_obs": obses_tp1, "dones": dones, "weights": np.ones_like(rewards)})
        return batch

    def opt_step(self, weights):
        self.set_weights(weights)
        samples = self.sample()
        results = self.asq_graph.opt_step(
            self.sess, samples["obs"], samples["actions"], samples["rewards"],
            samples["new_obs"], samples["dones"], samples["weights"])
        if self.config["prioritized_replay"]:
            new_priorities = (np.abs(results["td_error"]) + self.config["prioritized_replay_eps"])
            self.replay_buffer.update_priorities(samples["batch_indexes"], new_priorities)
        return self.get_weights()

    def set_global_timestep(self, global_timestep):
        self.global_timestep = global_timestep

    def get_weights(self):
        if self.model_idx is None:
            weights = self.asq_graph.get_target_q_func_vars(self.sess)
        else:
            weights = (self.model_idx, self.asq_graph.get_q_func_vars(self.sess))
        return weights

    def set_weights(self, weights, model_idx=None):
        self.asq_graph.set_target_q_func_vars(self.sess, weights, model_idx)
        if self.model_idx is None:
            return self.get_weights()

    def _step(self):
        action = self.asq_graph.act(self.sess, np.asarray(self.obs)[None], self.exploration.value(self.global_timestep))[0]
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

    def stats(self):
        n = self.config["smoothing_num_episodes"] + 1
        mean_episode_reward = round(np.mean(self.episode_rewards[-n:-1]), 5)
        mean_episode_length = round(np.mean(self.episode_lengths[-n:-1]), 5)
        exploration = self.exploration.value(self.global_timestep)
        return {
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
            "num_episodes": len(self.episode_rewards),
            "exploration": exploration,
            "local_timestep": self.local_timestep,
        }

    def save(self):
        ret = [
            self.exploration,
            self.episode_rewards,
            self.episode_lengths,
            self.saved_mean_reward,
            self.obs,
            self.global_timestep]
        if self.model_idx is not None:
            ret.append(self.prioritized_replay_beta_schedule)
            ret.append(self.replay_buffer)
            ret.append(self.local_timestep)

    def restore(self, data):
        self.exploration = data[0]
        self.episode_rewards = data[1]
        self.episode_lengths = data[2]
        self.saved_mean_reward = data[3]
        self.obs = data[4]
        self.global_timestep = data[5]
        if self.model_idx is not None:
            self.prioritized_replay_beta_schedule = data[6]
            self.replay_buffer = data[7]
            self.local_timestep = data[8]
