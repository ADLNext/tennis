# Tennis prediction

## What's in this repo

* The folder `notebooks` contains IPython notebooks with some POC/explorations:

- `tennis_ML.ipynb`: an IPython notebook with a comparison among different ML models; this is just a Proof of Concept and needs to be extended and carefully evaluated.
So far there is just a small study to predict a hypothetical Djokovic vs. Nadal match.

- `tennis_p2p.ipynb`: another IPython notebook with a brief dataset exploration on the point to point dataset.

- `tennis_simulation.ipynb`: a notebook exploring the possibility of modeling a tennis game as a stochastic simulation; the dataset is the same used in the `tennis_p2p.ipynb` notebook. WIP.

* `clf_comparison.py`: script extracted from the notebook with the comparison; script contains more ML models and a Deep Neural Network as well.

* `mlp_comparison.py`: same as `clf_comparison.py`, but comparing multiple MLPs configs up to 4 hidden layers.

* `tennis_clf.py`: this script uses the data preprocessing from the CNN approach with multiple configurations of MLPs.

* `tennis_proba.py`: moving from binary prediction to probability estimation.

### CNN for score prediction

This brave approach uses a Convolutional Neural Network to try and predict the scores of a tennis match.

The folder `support` contains support scripts to preprocess data; these scripts can be useful for different approaches as well.

The script `tennis_convolution.py` implements a CNN in tensorflow, including loss function and optimizer.
It's a work in progress and I still need to go through the math for the convolution and pooling layers.
