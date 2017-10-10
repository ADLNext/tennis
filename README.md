# Tennis prediction

## What's in this repo

- `tennis_ML.ipynb`: an IPython notebook with a comparison among different ML models; this is just a Proof of Concept and needs to be extended and carefully evaluated.
So far there is just a small study to predict a hypothetical Djokovic vs. Nadal match.

- `clf_comparison.py`: script extracted from the notebook with the comparison; script contains more ML models and a Deep Neural Network as well.

- `mlp_comparison.py`: same as `clf_comparison.py`, but comparing multiple MLPs configs up to 4 hidden layers.

- `tennis_p2p.ipynb`: another IPython notebook with a brief dataset exploration on the point to point dataset.

- `tennis_simulation.ipynb`: a notebook exploring the possibility of modeling a tennis game as a stochastic simulation; the dataset is the same used in the `tennis_p2p.ipynb` notebook. WIP.

### CNN for score prediction

This brave approach uses a Convolutional Neural Network to try and predict the scores of a tennis match.

The folder `support` contains support scripts to preprocess data; these scripts can be useful for different approaches as well.

The script `tennis_convolution.py` implements a CNN for regression in tensorflow, including loss function and optimizer.
It's a work in progress and I haven't been able to properly test it because of the computing power required.
