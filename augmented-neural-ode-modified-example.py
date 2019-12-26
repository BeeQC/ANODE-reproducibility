
import matplotlib.pyplot as plt
import torch

device = torch.device('cpu')

from experiments.dataloaders import ConcentricSphere
from torch.utils.data import DataLoader
from viz.plots import single_feature_plt

# Create training data in 2D, consisting of a concentric disk and an annulus
data_dim = 2
data_concentric = ConcentricSphere(data_dim, inner_range=(0., .8), outer_range=(.7, 1.5), 
                                   num_points_inner=2000, num_points_outer=2000)
dataloader = DataLoader(data_concentric, batch_size=64, shuffle=True)

# Visualize a batch of data (use a large batch size for visualization)
dataloader_viz = DataLoader(data_concentric, batch_size=256, shuffle=True)
for inputs, targets in dataloader_viz:
    break

single_feature_plt(inputs, targets,save_fig='olap_viz.png')

from anode.models import ODENet
from anode.training import Trainer

hidden_dim = 32

model = ODENet(device, data_dim, hidden_dim, time_dependent=True,
               non_linearity='relu')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

from viz.plots import get_feature_history

# Set up trainer
trainer = Trainer(model, optimizer, device)
num_epochs = 12

# Optionally record how the features evolve during training
visualize_features = True

if visualize_features:
    feature_history = get_feature_history(trainer, dataloader, inputs,
                                          targets, num_epochs)
else:
    # If we don't record feature evolution, simply train model
    trainer.train(dataloader, num_epochs)

from viz.plots import multi_feature_plt

multi_feature_plt(feature_history[::2], targets,save_fig='node_feats.png')

from viz.plots import trajectory_plt

# To make the plot clearer, we will use a smaller batch of data
for small_inputs, small_targets in dataloader:
    break

trajectory_plt(model, small_inputs, small_targets, timesteps=10,save_fig='node_trajectory.png')

from viz.plots import input_space_plt

input_space_plt(model,save_fig='node_space.png')

# Add 1 augmented dimension
anode = ODENet(device, data_dim, hidden_dim, augment_dim=1,
               time_dependent=True, non_linearity='relu')

optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)

# Set up trainer
trainer_anode = Trainer(anode, optimizer_anode, device)
num_epochs = 6

# Optionally record how the features evolve during training
visualize_features = True

if visualize_features:
    feature_history = get_feature_history(trainer_anode, dataloader, 
                                          inputs, targets, num_epochs)
else:
    # If we don't record feature evolution, simply train model
    trainer_anode.train(dataloader, num_epochs)

# Plot features and trajectories
from viz.plots import multi_feature_plt

multi_feature_plt(feature_history, targets,save_fig='anode_features.png')

from viz.plots import trajectory_plt

trajectory_plt(anode, small_inputs, small_targets, timesteps=10,save_fig='anode_trajectories.png')

from viz.plots import input_space_plt

input_space_plt(anode,save_fig='anode_input.png')

# Plot iteration by NFEs
fig,ax1=plt.subplots()
color = '#005AB5'
ax1.set_xlabel('Iterations (NODE)', color=color)
ax1.set_ylabel('NFEs')
ax1.plot(trainer.histories['nfe_history'], color=color)
ax1.tick_params(axis='x', labelcolor=color)
ax1.set_xlim(0, len(trainer.histories['nfe_history']) - 1)
ax2 = ax1.twiny()

color = '#DC3220'
ax2.set_xlabel('Iterations (ANODE)', color=color)
ax2.plot(trainer_anode.histories['nfe_history'], color=color)
ax2.tick_params(axis='x', labelcolor=color)
ax2.set_xlim(0, len(trainer_anode.histories['nfe_history']) - 1)
fig.tight_layout()
plt.show()

fig.savefig('nfe.png')

# Plot iteration by loss
fig,ax1=plt.subplots()
color = '#005AB5'
ax1.set_xlabel('Iterations (NODE)', color=color)
ax1.set_ylabel('Loss')
ax1.plot(trainer.histories['loss_history'], color=color)
ax1.tick_params(axis='x', labelcolor=color)
ax1.set_xlim(0, len(trainer.histories['loss_history']) - 1)

ax2 = ax1.twiny()

color = '#DC3220'
ax2.set_xlabel('Iterations (ANODE)', color=color)
ax2.plot(trainer_anode.histories['loss_history'], color=color)
ax2.tick_params(axis='x', labelcolor=color)
ax2.set_xlim(0, len(trainer_anode.histories['loss_history']) - 1)
fig.tight_layout()
plt.show()
fig.savefig('loss.png')