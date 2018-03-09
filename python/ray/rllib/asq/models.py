import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_q_network(num_actions, config, inputs):
    dueling = config["dueling"]
    hiddens = config["hiddens"]
    filters = config["conv_filters"]
    with tf.name_scope("vision_net"):
        for i, (out_size, kernel, stride) in enumerate(filters, 1):
            inputs = layers.conv2d(inputs, out_size, kernel, stride, scope="conv{}".format(i))
        visual_features = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])

    with tf.variable_scope("action_value"):
        action_out = visual_features
        for hidden in hiddens:
            action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
        action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

    if dueling:
        with tf.variable_scope("state_value"):
            state_out = visual_features
            for hidden in hiddens:
                state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
        action_scores_mean = tf.reduce_mean(action_scores, 1)
        action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
        return state_score + action_scores_centered
    else:
        return action_scores


def build_action_network(num_actions, inputs, q_values, eps):
    batch_size = tf.shape(inputs)[0]
    random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
    chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    deterministic_actions = tf.argmax(q_values, axis=1)
    return tf.where(chose_random, random_actions, deterministic_actions)


def huber_loss(x, delta=1.0):
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta))


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    return [(tf.clip_by_norm(grad, clip_val), var) for (grad, var) in gradients]


class ASQGraph(object):
    def __init__(self, env, config, ensemble_size, trainable=True):
        self.ensemble_size = ensemble_size
        num_actions = env.action_space.n
        with tf.variable_scope("ASQGraph"):
            self.input_observation = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape)
            self.eps = tf.placeholder(tf.float32, (), name="eps")
            with tf.variable_scope("target_q_func"):
                q_tp1_ensemble = []
                self.target_q_func_vars = []
                self.target_q_func_ph = []
                self.target_q_func_assign = []
                for model_idx in range(ensemble_size):
                    with tf.variable_scope("model{}".format(model_idx)) as scope:
                        q_tp1_ensemble.append(build_q_network(num_actions, config, self.input_observation))
                        self.target_q_func_vars.append(scope.trainable_variables())
                        self.target_q_func_ph.append([tf.placeholder(v.dtype, v.shape, v.name[v.name.rindex("/") + 1:-2] + "_ph")
                                                      for v in self.target_q_func_vars[-1]])
                        self.target_q_func_assign.append([tf.assign(v, ph) for v, ph in zip(self.target_q_func_vars[-1], self.target_q_func_ph[-1])])
                q_tp1 = tf.reduce_mean(tf.stack(q_tp1_ensemble, 2), 2)
                self.output_action = build_action_network(num_actions, self.input_observation, q_tp1, self.eps)
            if trainable:
                self.obs_t = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape)
                self.act_t = tf.placeholder(tf.int32, [None], name="action")
                self.rew_t = tf.placeholder(tf.float32, [None], name="reward")
                self.obs_tp1 = self.input_observation
                self.done_mask = tf.placeholder(tf.float32, [None], name="done")
                self.importance_weights = tf.placeholder(tf.float32, [None], name="weight")
                with tf.variable_scope("q_func") as scope:
                    q_t = build_q_network(num_actions, config, self.obs_t)
                    self.q_func_vars = scope.trainable_variables()
                q_t_selected = tf.reduce_sum(q_t * tf.one_hot(self.act_t, num_actions), 1)
                q_tp1_best = tf.reduce_max(q_tp1, 1)
                q_tp1_best_masked = (1.0 - self.done_mask) * q_tp1_best
                q_t_selected_target = self.rew_t + config["gamma"] * q_tp1_best_masked
                self.td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
                errors = huber_loss(self.td_error)
                self.loss = tf.reduce_mean(self.importance_weights * errors)
                optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"])
                if config["grad_norm_clipping"] is not None:
                    grads_and_vars = minimize_and_clip(optimizer, self.loss, var_list=self.q_func_vars, clip_val=config["grad_norm_clipping"])
                else:
                    grads_and_vars = optimizer.compute_gradients(self.loss, var_list=self.q_func_vars)
                self.opt_op = optimizer.apply_gradients(grads_and_vars)

    def act(self, sess, obs, eps):
        return sess.run(
            self.output_action,
            feed_dict={
                self.input_observation: obs,
                self.eps: eps,
            })

    def opt_step(self, sess, obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights):
        return sess.run({
            'td_error': self.td_error,
            'opt_step': self.opt_op},
            feed_dict={
                self.obs_t: obs_t,
                self.act_t: act_t,
                self.rew_t: rew_t,
                self.obs_tp1: obs_tp1,
                self.done_mask: done_mask,
                self.importance_weights: importance_weights
            })

    def get_q_func_vars(self, sess):
        return sess.run(self.q_func_vars)

    def get_target_q_func_vars(self, sess):
        return sess.run(self.target_q_func_vars)

    def set_target_q_func_vars(self, sess, weights, model_idx=None):
        if model_idx is None:
            for i in range(self.ensemble_size):
                sess.run(self.target_q_func_assign[i], feed_dict={v: w for v, w in zip(self.target_q_func_ph[i], weights[i])})
        else:
            sess.run(self.target_q_func_assign[model_idx], feed_dict={v: w for v, w in zip(self.target_q_func_ph[model_idx], weights)})
