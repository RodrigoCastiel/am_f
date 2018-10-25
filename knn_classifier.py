"""
This modules defines class KNNClassifier, a parametric KNN estimator for general
n-dimensional datasets.
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
import heapq
from data_loader import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, K = 7):
    self.K = K
    self.x_train = []
    self.w_train = []
    self.p_w = []

  def fit(self, x_train, w_train):
    """
    Stores training points *x_train* and their correponsindg labels *w_train*,
    and estimates the a prior probabilities p(w_i) for each class w_i.
    """
    # Store examples.
    self.x_train = x_train
    self.w_train = w_train

    # Estimate a prior probabilities p(w_i) for each class w_i.
    x_groups = DataLoader.group_by_label(x_train, w_train)
    self.p_w = np.array(list(map(
      lambda x_train_k: len(x_train_k)/len(x_train),
      x_groups,
    )))

    return self

  def predict(self, x_set):
    """
    Runs KNN prediction/estimation for each point x in x_set.
    Returns an array containing the predicted classes for each input point.
    """
    return np.array(list(map(
      lambda x: self.classify(x),
      x_set,
    )))

  def compute_a_posteriori(self, x):
    # TODO(RodrigoCastiel): estimate p(wi|x) through the number of votes in KNN.
    pass

  def classify(self, x):
    """
    Classifies a single data-point. Returns the predicted integer label.
    """
    # Evaluate the distance L2 of x to all training points.
    dist  = np.linalg.norm(x - self.x_train, axis=1)

    # Compute indices of the k nearest points (with respect to training_data).
    nearest_k_indices = np.argpartition(dist, kth=self.K)

    # Majority wins.
    count = {}
    for label in [self.w_train[k] for k in nearest_k_indices]:
      count[label] = count.get(label, 0) + 1
    return max(count, key=count.get)

  def evaluate(self, x_test, w_test):
    """
    Returns a tuple (num_correct_predictions, accuracy), meaning the number of
    correct estimations/predictions and the accuracy.
    """
    w_est = self.predict(x_test)
    num_correct_predictions = np.sum(w_est == np.array(w_test))
    accuracy = num_correct_predictions/float(len(w_est))
    return (num_correct_predictions, accuracy)