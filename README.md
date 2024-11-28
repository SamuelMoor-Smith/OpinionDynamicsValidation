# OpinionDynamicsValidation

## Models

The models are all given in the model folder. Currently implemented are Deffuant, HK-averaging (with different means) and Carpentras

## Datasets

The dataset.py file allows the user to create different snapshot datasets via various methods. The main one is with an initial opinion distribution and a model for updating

## Experiments

The experiments folder contains the plot-generating experiment code. Right now one experiment tests the optimizer on no-noise updates and the other looks at the optimizers capability with various amounts of noise (standardized with the explained variance of that amount of noise)

## Utils

Utils contains all utility code ie to calculate differences, add noise, optimizer calls, plotting, and the random generation code.

## Plots

Contained in plot folder under model and no_noise/noise

## Results

Contains extra "write_to_file" info - mainly whether optimizer does better than the true parameters or not.
