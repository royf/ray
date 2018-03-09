from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.rllib.optimizers.optimizer import Optimizer
from ray.rllib.utils.timer import TimerStat


class EnsembleOptimizer(Optimizer):
    def _init(self):
        self.step_timer = TimerStat()
        self.sync_timer = TimerStat()
        self.opts_per_step = self.config.get("opts_per_step", 100)

    def step(self):
        ensemble_weights = ray.put(self.local_evaluator.get_weights())
        task_queue = []
        num_opts = 0
        for e in self.remote_evaluators:
            task_queue.append((e.opt_step.remote(ensemble_weights), e))
            num_opts += 1
            if num_opts >= self.opts_per_step:
                break

        # Note: can't use wait: https://github.com/ray-project/ray/issues/1128
        while num_opts < self.opts_per_step:
            with self.step_timer:
                while True:
                    w, e = task_queue.pop(0)
                    ready, remaining = ray.wait([w], timeout=100)
                    if remaining:
                        task_queue.append((remaining[0], e))
                    else:
                        task_queue.append((e.opt_step.remote(ensemble_weights), e))
                        num_opts += 1
                        model_idx, model_weights = ray.get(ready[0])
                        break
            with self.sync_timer:
                ensemble_weights = self.local_evaluator.set_weights(model_weights, model_idx)

    def stats(self):
        return {
            "step_time_ms": round(1000 * self.step_timer.mean, 3),
            "sync_time_ms": round(1000 * self.sync_timer.mean, 3),
        }
