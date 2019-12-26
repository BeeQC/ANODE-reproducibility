# ANODE-reproducibility
This repo contains the code we used as part of the [Reproducibility Challenge @ NeurIPS 2019](https://reproducibility-challenge.github.io/neurips2019/) to reproduce the paper [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681), by Dupont, Arnaud & Teh (2019). 

Most of the scripts contained here have minor modifications to their code, which can be found here: [Augmented Neural ODEs](https://github.com/EmilienDupont/augmented-neural-odes). Their code is well commented and easy to implement, please refer to their page for details.

## Readme for this repo ##
The Results folder contains the saved experiments that we performed along with the code files to plot these experiments.
The config files are contained within each data folder.

To rerun RK4 experiments please comment lines 151-158 in ```models.py```, and uncomment lines 159-166

The jupyter notebook ```num_filters_fig.ipynb``` was used for the hyperparameter search.


## Modifications from [Augmented Neural ODEs](https://github.com/EmilienDupont/augmented-neural-odes) ##
Below you will find any modifications we have made to the original files. Only files with modifications appear on the list below. The code contained in folders ```Plotting``` and ```Results``` were partially based on several ANODE files and to perform our experiments.

### Main folder ###
- ```augmented-neural-ode-example.ipnyb```: modifications to number of epochs to both models and number of points and ranges in concentric sphere model  
- ```augmented-neural-ode-example.py```: based on the notebook above, contains modifications to save figures and to plot NFE/Loss vs. iteration (and the modifications listed above)
- ```make_moons_modifications```: this script contains modifications to the code that were removed since we could not get it to work. We attempted to implement ANODE to classify the [Sklearn Moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) toy dataset. To use this portion of the code, ```sklearn``` should be added to the list of requirements.

### ```anode``` folder ###

- ```models.py```: added RK4 experiments (i.e. ODE solver with number of evaluations fixed to 4)
- ```training.py```: modified to return accuracy

### ```experiments``` folder ###

- ```experiments.py```: modified to return accuracy and to run 1D experiments
- ```experiments_img.py```: modified to return accuracy

### ```viz``` folder ###
- ```plots.py```: modified image dimensions and resolution, removed calls to ```set_aspect()``` 
