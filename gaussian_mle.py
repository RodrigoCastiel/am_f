"""
This module defines class GassianMLE, a maximum likelihood classifier for
general datasets. It models the underlying generator pdf as a multivariate
normal distribution.
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from data_loader import DataLoader

class GaussianMLE:
  def __init__(self):
    # Estimated a prior probabilities.
    self.p_w = []

  def get_a_priori_probabilities(self):
    """
    Returns the list of prior probabilities p(w_i) for each class w_i.
    """
    return self.p_w

  def train(self, x_train, w_train, num_classes):
    """
    Estimates a prior probabilities p(w_i), the mean_i and the cov_i matrix for
    each class in the training dataset. That is, from (x_train, w_train).
    Once trained, you can call classify() to predict the class/label
    for a given feature vector.
    """
    # Break down dataset into smaller groups sharing the same label.
    int_labels = list(range(num_classes))
    groups = DataLoader.group_by_label(x_train, w_train, int_labels)

    # Estimate a prior probabilities p(w_i) for each class w_i.
    self.p_w = list(map(lambda x_train_k: len(x_train_k)/len(x_train), groups))

  def classify(self, x_sample):
    pass

  @staticmethod
  def estimate_gaussian_parameters(self, x_train_i):
    pass