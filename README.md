## PINN - Gravitational Waves

This repository contains some useful code used to recover the orbits of binary black holes system from its gravitational waveform measurements. Experiments are performed using Julia code but some visualizations, analysis and data extraction are done in Python.
# Recovering BBH equation of motions from waveform measurements using Physics Informed Neural Networks (PINNs)

The purpose of this repository is to be used as a template to be initiated into the problem of recovering BBH equation of motions from waveform measurements using Physics Informed Neural Networks (PINNs) in Julia. Let's present how it is divided:

### Data

This folder is currently created as an example but note that strictly this should not be uploaded to the repository. 

Two folders (input and output) are contained in data folder. Consequent folders are contained here for each experiment. For example: *case_2/schwarzschild/models/test_1_cos*. Inside this folder, one will find:

- solutions: weights of the neural networks
- predictions: orbit, equation of motion and waveform predictions as well as true values
- train_img_for_gif: images of a train and a test waveform for each training epoch

Also, one will find a table in *case_2/schwarzschild/metrics/* called *losses.csv* which keeps record of results for different models in *case_2/schwarzschild/models/*

### Code

Two parts are distinguished here: *src/utils*, which contains the different functions used to develop the different experiments, and *src/processing*, which contains subfolders with the scripts to perform experiments. Under *src/processing*, there is a folder where experiments are referred called *src/processing/experiments*. Moreover, there is a script to get SXS:BBH data under *src/processing/collect_data/CollectSXSData.py*. 

There are two main experiments:

- **Case 1: Extreme Mass Ratio (EMR) systems**. Both **Schwarzschild** and **Kerr** metrics are proposed resulting in two variants of the same execise.
- **Case 2: non-EMR systems** of equal masses and non-eccentric orbits on a model based only on **Schwarzschild** metric.

Each block contains 2 Jupyter notebooks:
- **Training Notebook**: it generates the train and test datasets, trains the PINN and saves the model weights. It had the option to get some intermediate plots and saves a register of loss function for train and test data.
- **Exploration Notebook**: it loads the trained model weights and predicts waveforms. It also incorporates some basic plots

Each block also includes one folder named */old/* which contains scripts of old experiments which nowdays don't work due to the repo refactor. In a further update, this will be eliminated or incoporated in the src/utils code.

**NOTE**: some techniques discussed at the work are not directly implemented in the repo but can be done adjusting some parameters or comenting some code. 


### Coments on what it is

- Only real components of waveforms are taken into account in the analysis since it is assumed that they are representative to calculate the waveform error in the loss function to train the neural network.
- Code is designed to enable training on datasets of several items. However, for simplicity notebooks 'as-they-are' only include one item for dataset. One for train and other for test purposes.
- Current base model for the PINN in the Kerr metric is the exact solution. This must be modified to work on experiments since the current run of the notebook does not shown any remarkable results.
