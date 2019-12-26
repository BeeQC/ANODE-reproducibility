'''Attempt at adding new toy dataset did NOT work. Here are the modifications we made per file,
which can be added back at the lines indicated'''

'''
dataloaders.py
'''
# Add on line 180 (after all classes have been specified)
class Moons(Dataset):
    def __init__(self, num_points, noise_scale=None, dim=2):
        self.num_points = num_points
        self.noise_scale = noise_scale
        self.data, self.targets = make_moons(n_samples=self.num_points,
                                             noise=noise_scale)
        self.data = torch.from_numpy(self.data)
        self.dim = dim
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.num_points

# Add on line 9 (after all other libraries were imported)
from sklearn.datasets import make_moons

'''
experiments.py
'''
# Add to line 11 (where datasets are imported)
from experiments.dataloaders import ConcentricSphere, ShiftedSines, Data1D, Moons

# Add at line 68 (after the other dataset types have been specified)
        elif dataset["type"] == "moons":
            data_object = Moons(dim=data_dim,
                                num_points=dataset["num_points"],
                                noise_scale=dataset["noise_scale"])

'''
config.json
'''
# Add at line 5 (after datasets is specified)
    {
	  "type": "moons",
	  "num_points": 1000,
	  "noise_scale": 0.2
	}