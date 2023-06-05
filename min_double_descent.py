"""
A minimal example of double descent using ridge regression trained
on tanh random features. The dataset is a linearly separable
classification problem with 2 features and 2 classes.
"""

import matplotlib.pyplot as plt
import pandas as pd
from numpy import argmax, eye, linalg, mean, random, tanh, unique
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as split
from tqdm import trange

seed = 42
N = 10  # Number of training/test samples
P_max = N * 5  # Maximum number of random features
P_step = P_max // 50  # Step size for number of random features
d = 2  # Number of features
num_trials = 10000  # Number of trials to average over

x, y = make_classification(
    n_samples=N * 2,
    n_features=d,
    n_informative=d,
    n_redundant=0,
    flip_y=0,
    class_sep=1.5,
    random_state=seed,
)
y = eye(unique(y).shape[0])[y]  # One-hot encode the labels
x_tr, x_te, y_tr, y_te = split(x, y, test_size=0.5, random_state=seed)


def cond_number(x):
    """Compute the condition number of a matrix."""
    _, s, _ = linalg.svd(x)
    return s[0] / (s[-1] + 1e-8)  # Add division by zero


# Generate a random matrix with iid Gaussian entries
random_matrix = lambda d, P: random.normal(loc=0, scale=1 / d**0.5, size=(d, P))
# Compute the random tanh features
random_features = lambda x, W0: tanh(x @ W0)
# Compute the mean squared error
mse_loss = lambda y, y_pred: mean((y - y_pred) ** 2)
# Compute the ridge regression solution
ridge = lambda X, y, a=1e-8: linalg.inv(X.T @ X + a * eye(X.shape[1])) @ X.T @ y


def main(x_tr, y_tr, x_te, y_te, P_max, P_step, num_trials):
    output = []
    for _ in trange(num_trials):
        W0_ = random_matrix(d, P_max)
        for p in range(P_max, 0, -P_step):
            W0 = W0_[:, :p]  # Reuse part of the random matrix for efficiency
            # Compute the random features
            z_tr, z_te = random_features(x_tr, W0), random_features(x_te, W0)
            W1 = ridge(z_tr, y_tr)  # Compute the ridge regression solution
            tr_pred, te_pred = z_tr @ W1, z_te @ W1  # Compute the predictions
            output.append(
                {
                    "p_over_n": p / N,
                    "tr_acc": mean(argmax(tr_pred, 1) == argmax(y_tr, 1)),
                    "te_acc": mean(argmax(te_pred, 1) == argmax(y_te, 1)),
                    "tr_loss": mse_loss(y_tr, tr_pred),
                    "te_loss": mse_loss(y_te, te_pred),
                    "cond": cond_number(z_tr),
                    "W1_norm": linalg.norm(W1, ord=2),
                }
            )
    return pd.DataFrame(output).groupby("p_over_n").mean().reset_index()


def plot(output):
    _, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    # Plot error
    axes[0].plot(output["p_over_n"], 1 - output["tr_acc"], label="Train")
    axes[0].plot(output["p_over_n"], 1 - output["te_acc"], label="Test")
    axes[0].axvline(x=1, color="r", linestyle="--")
    axes[0].set(xlabel=r"$P/N$", ylabel="Error Rate")
    axes[0].legend()
    # Plot loss
    axes[1].plot(output["p_over_n"], output["tr_loss"], label="Train")
    axes[1].plot(output["p_over_n"], output["te_loss"], label="Test")
    axes[1].axvline(x=1, color="r", linestyle="--")
    axes[1].set(xlabel=r"$P/N$", ylabel="MSE Loss", yscale="log")
    axes[1].legend()
    # Plot condition number
    axes[2].plot(output["p_over_n"], output["cond"])
    axes[2].axvline(x=1, color="r", linestyle="--")
    axes[2].set(xlabel=r"$P/N$", ylabel=r"$\sigma_{\max}/\sigma_{\min}$", yscale="log")
    # Plot norm of W1
    axes[3].plot(output["p_over_n"], output["W1_norm"])
    axes[3].axvline(x=1, color="r", linestyle="--")
    axes[3].set(xlabel=r"$P/N$", ylabel=r"$\|W_1\|_2$", yscale="log")
    plt.tight_layout()
    plt.savefig("min_double_descent.pdf", bbox_inches="tight")
    plt.show()


plot(main(x_tr, y_tr, x_te, y_te, P_max, P_step, num_trials))
