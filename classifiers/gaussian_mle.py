"""
This module defines class GassianMLE, a maximum likelihood classifier for
general datasets. It models the underlying generator pdf as a multivariate
normal distribution with uncorrelated feature dimensions.
In other words, x ~ N_i(mi, diag(sigma)) for each class i.

Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from core.data_loader import DataLoader
from core.committee_classifier_base import CommitteeClassifierBase

class GaussianMLE(CommitteeClassifierBase):
  def __init__(self):
    # Prior probabilities p(w_i) for each class w_i.
    self.p_w = []
    # Estimated means mi_i [N-array] for each class w_i.
    self.mu = []
    # Estimated variances sigma_i for each class w_i.
    self.sigma = []
    # Precomputed inverse variances (sigma_i^-1).
    self.inv_sigma = []
    # Precomputed amplitudes for the gaussian pdfs.
    self.amplitudes = []

  def fit(self, x_train, w_train):
    """
    Estimates a prior probabilities p(w_i), the mean i and the variance i for
    each class in the training dataset. That is, from (x_train, w_train).
    Once trained, you can call classify() to predict the class/label
    for a given feature vector.
    Note: all classes must have at least one sample.
    """
    # Break down dataset into smaller groups sharing the same label.
    x_groups = DataLoader.group_by_label(x_train, w_train)

    # Estimate a prior probabilities p(wi) for each class wi.
    self.p_w = DataLoader.compute_a_priori(w_train)

    # Estimate mean and [diagonal] variances for each class w_i.
    # Pattern Classification (Second Edition), Section 3.2.3.
    self.mu = np.array(list(map(
      lambda x_train_i: np.mean(x_train_i, axis=0),
      x_groups,
    )))
    self.sigma = np.array(list(map(
      lambda i: np.mean((x_groups[i] - self.mu[i])**2, axis=0),
      range(len(x_groups)),
    )))

    # For the sake of optimization, we may precompute some constants in the 
    # gaussian pdf equations - the amplitudes and the inverse of sigma.
    epsilon = 1e-6
    n = len(self.mu)
    pi_const = np.power(2*np.pi, n/2.0)
    det_sigma = np.abs(np.product(self.sigma, axis=1))
    self.inv_sigma = 1.0/(self.sigma + epsilon)
    self.amplitudes = 1.0/(pi_const * np.sqrt(det_sigma + epsilon))

    return self

  def predict(self, x_set):
    """
    Runs prediction/estimation for each point x in x_set. That is,
    chooses the class that maximuzes the scaled a posterior probability:
        wj = argmax_wi { p(x | wi) p(wi) }
        (excluding p(x) from the denomunator).
    Returns an array containing the predicted classes for each input point.
    """
    def classify(x):
      p_x_wi = self.compute_likelihoods(x)
      return np.argmax(p_x_wi * self.p_w)

    return np.array(list(map(classify, x_set)))

  def compute_a_priori(self):
    return self.p_w

  def compute_a_posteriori(self, x):
    """
    Computes the a posteriori probability p(wi|x) for each class wi, given by:
      p(wi|x) = [p(x|wi)p(wi)] / sum_k(p(x|wk)p(wk))
    Or simply:
      p(wi|x) = p(x|wi)p(wi) / p(x).
    """
    # Likelihood: p(x|wi).
    p_x_wi = self.compute_likelihoods(x)
    # A priori probability: p(wi).
    p_wi = self.p_w
    # Joint proability p(x && wi).
    p_x_and_wi = p_x_wi * p_wi

    return p_x_and_wi / np.sum(p_x_and_wi)

  def compute_likelihoods(self, x):
    """
    Computes the likelihood of x for each class w_k, using the gaussian pdf.
    Returns the numpy array [p(x|w1), p(x|w2), ...].
    """
    A = self.amplitudes
    mu, sigma_inv = self.mu, self.inv_sigma
    return A * np.exp(-0.5 * np.sum(sigma_inv * (x - mu)**2, axis=1))

  def evaluate(self, x_test, w_test):
    """
    Returns a tuple (num_correct_predictions, accuracy), meaning the number of
    correct estimations/predictions and the accuracy.
    """
    w_est = self.predict(x_test)
    num_correct_predictions = np.sum(w_est == np.array(w_test))
    accuracy = num_correct_predictions/float(len(w_est))
    return (num_correct_predictions, accuracy)
