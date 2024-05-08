## PINN - Gravitational Waves

This repository contains some useful code used to recover the orbits of binary black holes system from its gravitational waveform measurements. Experiments are performed using Julia code but some visualizations, analysis and data extraction are done in Python.

The repository has two main parts: data and code. Its purpose is to be used as a template to be initiated into the problem.

### Data

Two folders (input and output) are contained in data folder. Consequent folders are contained here for each experiment. For example: experiment 1 -

### Source Code

Two parts are distinguished here: utils, which contains the different pieces of code used to develop the different experiments, and processing, which contains subfolders with the scripts to perform experiments.

#### Processing

##### Schwarzschild

Here experiments contained with Schwarzschild metric are peformed regarding to 2 cases: EMR (Extreme Mass Ratio) case and non-EMR. Inside non-EMR, we found: Equal masses non-eccentric orbit case, equal masses eccentric orbit case and non-equal masses eccentric orbit case.

##### Kerr

Here experiments contained with kerr metric are performed. Only EMR case is analyzed.
