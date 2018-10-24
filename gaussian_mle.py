"""
This module defines class GassianMLE, a maximum likelihood classifier for
general datasets. It models the underlying generator pdf as a multivariate
normal distribution with uncorrelated feature dimensions.
In other words, x ~ N_i(mi, diag(sigma)) for each class i.

Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from data_loader import DataLoader

class GaussianMLE:
  def __init__(self):
    # List of prior probabilities p(w_i) for each class w_i.
    self.p_w = []
    # List of estimated means mi_i [N-array] for each class w_i.
    self.mi = []
    # List of estimated variances sigma_i for each class w_i.
    self.sigma = []
    # List of precomputed inverse variances (sigma_i^-1).
    self.inv_sigma = []
    # List of precomputed amplitudes for the gaussian pdfs.
    self.amplitudes = []

  def train(self, x_train, w_train):
    """
    Estimates a prior probabilities p(w_i), the mean i and the variance i for
    each class in the training dataset. That is, from (x_train, w_train).
    Once trained, you can call classify() to predict the class/label
    for a given feature vector.
    """
    # Break down dataset into smaller groups sharing the same label.
    num_classes = len(np.unique(w_train))
    int_labels = list(range(num_classes))
    x_groups = DataLoader.group_by_label(x_train, w_train, int_labels)

    # Estimate a prior probabilities p(w_i) for each class w_i.
    self.p_w = list(map(
      lambda x_train_k: len(x_train_k)/len(x_train),
      x_groups,
    ))

    # Estimate mean and [diagonal] variances for each class w_i.
    # Pattern Classification (Second Edition), Section 3.2.3.
    self.mi = np.array(list(map(
      lambda x_train_i: np.mean(x_train_i, axis=0),
      x_groups,
    )))
    self.sigma = np.array(list(map(
      lambda i: np.mean((x_groups[i] - self.mi[i])**2, axis=0),
      range(len(x_groups)),
    )))

    # For the sake of optimization, we may precompute some constants in the 
    # gaussian pdf equations - the amplitudes and the inverse of sigma.
    epsilon = 1e-6
    det_sigma = np.abs(np.product(self.sigma, axis=1))
    self.inv_sigma = 1.0/(self.sigma + epsilon)
    self.amplitudes = 1.0/np.sqrt(det_sigma + epsilon)

  def predict(self, x_set):
    """
    Runs prediction/estimation for each point x in x_set. That is,
    chooses the class that maximizes the scaled a posterior probability:
        wj = argmax_wi { p(x_sample | w_i) p(w_i) }
        (excluding p(x) from the denominator).
    Returns an array containing the predicted classes for each input point.
    """
    def classify(x_sample):
      p_wi_x = self.compute_likelihoods(x_sample)
      return np.argmax(p_wi_x * self.p_w)

    return np.array(list(map(
      lambda x: classify(x),
      x_set,
    )))

  def evaluate(self, x_test, w_test):
    """
    Returns a tuple (num_correct_predictions, accuracy), meaning the number of
    correct estimations/predictions and the accuracy.
    """
    w_est = self.predict(x_test)
    num_correct_predictions = np.sum(w_est == np.array(w_test))
    accuracy = num_correct_predictions/float(len(w_est))
    return (num_correct_predictions, accuracy)

  def compute_likelihoods(self, x):
    """
    Computes the likelihood p(x|w_k) for each class w_k.
    It does NOT consider the constant (2pi)^(-n/2) in the gaussian formula.
    Returns the numpy array [p(x|w1), p(x|w2), ...].
    """
    A = self.amplitudes
    mi, sigma_i_inv = self.mi, self.inv_sigma
    return A * np.exp(-0.5 * np.sum(sigma_i_inv * (x - mi)**2, axis=1))
