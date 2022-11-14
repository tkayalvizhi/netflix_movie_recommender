"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, _ = X.shape
    K, d = mixture.mu.shape

    post = np.zeros((n, K))

    for i in range(n):
        x = X[i, :]
        for j in range(K):
            mu = mixture.mu[j, :]
            var = mixture.var[j] * np.identity(d)
            p_x_Cj = np.exp(-(1/2) * (x - mu) @ np.linalg.inv(var) @ (x - mu).T) / np.sqrt(((2 * np.pi) ** d) *
                                                                                           np.linalg.det(var))
            post[i][j] = mixture.p[j] * p_x_Cj

    log_likelihood = np.sum(np.log(np.sum(post, axis=1)))
    post = post / np.sum(post, axis=1, keepdims=True)

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    p = np.sum(post, axis=0) / np.sum(post)
    # ML estimates for mu_j
    mu = np.zeros((K, d))
    var = np.zeros((K,))

    for j in range(K):
        post_j = post[:, j][:, np.newaxis]

        mu[j, :] = np.sum(X * post_j, axis=0) / np.sum(post[:, j])
        var[j] = post_j.T @ np.sum((X - mu[j, :]) ** 2, axis=1)/ (d * np.sum(post[:, j]))

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_LL = None
    new_LL = None

    while (old_LL is None) or (new_LL - old_LL > (1e-6 * abs(new_LL))):
        old_LL = new_LL
        post, new_LL = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, new_LL