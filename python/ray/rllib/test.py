import ray
from ray.tune import register_trainable, grid_search, run_experiments

def my_func(config, reporter):
    import time, numpy as np
    i = 0
    while True:
        reporter(timesteps_total=i, mean_accuracy=i ** config["alpha"])
        i += config["beta"]
        time.sleep(.01)

register_trainable("my_func", my_func)

ray.init()
run_experiments({
    "my_experiment": {
        "run": "my_func",
        "resources": { "cpu": 1, "gpu": 0 },
        "stop": { "mean_accuracy": 100 },
        "config": {
            "alpha": grid_search([0.2, 0.4, 0.6]),
            "beta": grid_search(["a", "b"*400]),
        },
    }
})

