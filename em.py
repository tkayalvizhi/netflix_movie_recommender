"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, _ = mixture.mu.shape

    f = np.zeros((n, K), dtype=np.float64)

    for user in range(n):
        x = X[user, :]

        # indices of movies rated by user
        Cu = np.where(abs(x) >= 1e-12)

        # movies rated by user
        x_Cu = x[Cu]

        # number of movies rated by user
        d = x_Cu.shape[0]

        for j in range(K):
            # mean of spherical gaussian model j
            mu = mixture.mu[j, :][Cu]

            #  variance of spherical gaussian model j
            var = np.full(d, mixture.var[j], dtype=np.float64)

            # log conditional probability: probability of user given the spherical gaussian model j
            log_p_x_Cj = np.sum(-1/2 * np.log(2 * np.pi * var) - 1/(2*var) * (x_Cu - mu) ** 2)
            # log_p_x_Cj = np.sum(np.log(1 / (2 * np.pi * var) ** (1/2) * np.exp(-1/(2 * var) * (x_Cu - mu) ** 2)))

            # lof probability that user rates said movies and the user belongs to gaussianl model j
            # log P(A and B) -  intersection
            f[user][j] = np.log(mixture.p[j]) + log_p_x_Cj

    log_likelihood = np.sum(logsumexp(f, axis=1))
    log_post = f - logsumexp(f, axis=1, keepdims=True)

    return np.exp(log_post), log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    p = np.sum(post, axis=0) / np.sum(post)

    # ML estimates for mu_j
    mu = mixture.mu
    var = np.zeros((K,), dtype=np.float64)

    del_u = np.where(abs(X) >= 1e-12, 1, 0)
    count = np.sum(del_u, axis=1)

    for j in range(K):
        post_j = post[:, j][:, np.newaxis]

        denom = np.sum(post_j * del_u, axis=0)
        update_filter = np.where(denom >= 1)

        mu[j, :][update_filter] = (np.sum(X * post_j, axis=0) / denom)[update_filter]

        tiled_mu = np.tile(mu[j, :], (n, 1))

        var[j] = post_j.T @ np.sum((X * del_u - tiled_mu * del_u) ** 2, axis=1) / (np.sum(post[:, j] * count))
        if var[j] < min_variance:
            var[j] = min_variance

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
        mixture = mstep(X, post, mixture)

    return mixture, post, new_LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post, _ = estep(X, mixture)

    X_filled = X.copy()
    missing_indices = np.array(np.where(abs(X_filled) <= 1e-6)).T

    for user, movie in missing_indices:
        post_user = post[user, :]
        mu = mixture.mu[:, movie]

        X_filled[user, movie] = np.sum(mu * post_user)

    return X_filled



