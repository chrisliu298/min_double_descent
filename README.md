# Minimal Double Descent

This repository contains the code for demonstrating double descent on an extremely simple binary classification dataset with 10 samples, each with 2 features. The model uses random tanh features with ridge regression.

`min_double_descent.py` produces [a figure](min_double_descent.pdf) showing the peaking phenomenon in test error, test loss, condition number of the random features, and the norm of the weights.
