from experience_dataset import ExperienceDataset
from fast_cts import CTSDensityModel


class DensityModel(object):
    def __init__(self, num_actions=None):
        self.model_actions = num_actions is not None
        if self.model_actions:
            self.models = [CTSDensityModel() for _ in range(num_actions)]
        else:
            self.model = CTSDensityModel()

    def train(self, dataset, num_samples, idx_range=None):
        for _ in range(num_samples):
            sample = dataset.sample(1, idx_range)
            o = sample['observations'][0]
            a = sample['actions'][0]
            if self.model_actions:
                self.models[a].update(o[:, :, 0])
            else:
                self.model.update(o[:, :, 0])

    def test(self, dataset, num_samples, idx_range=None):
        sample = dataset.sample(num_samples, idx_range)
        if self.model_actions:
            return sum(self.models[sample['actions'][i]].log_prob(sample['observations'][i][:, :, 0]) for i in range(num_samples))
        else:
            return sum(self.model.log_prob(o[:, :, 0]) for o in sample['observations'])


def main():
    dataset = ExperienceDataset({"pong_42x42_experiences_19.9rew.json": .1})
    models = []
    for idx_range in [(0., .1), (.9, 1.)]:
        model = DensityModel(6)
        model.train(dataset, 100000, idx_range)
        models.append(model)
    scores = [[model.test(dataset, 1000, idx_range) for idx_range in [(i / 10., (i + 1) / 10.) for i in range(10)]] for model in models]
    print(scores)
