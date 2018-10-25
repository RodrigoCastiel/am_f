"""
This modules defines class KNNClassifier, a parametric KNN estimator for general
n-dimensional datasets.
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
import heapq
from data_loader import DataLoader
from committee_classifier_base import CommitteeClassifierBase

class KNNClassifier(CommitteeClassifierBase):
  def __init__(self, K = 3):
    self.K = K
    self.x_train = []
    self.w_train = []
    self.p_w = []
    self.num_classes = 0

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
    self.num_classes = len(self.p_w)

    return self

  def predict(self, x_set):
    """
    Runs KNN prediction/estimation for each point x in x_set.
    Returns an array containing the predicted classes for each input point.
    """
    def classify(x):
      # Pick top-voted label among the k nearest neighbors.
      label_votes = self.knn_label_votes(x)
      return max(label_votes, key=label_votes.get)

    return np.array(list(map(classify, x_set)))

  def compute_a_priori(self):
    return self.p_w

  def compute_a_posteriori(self, x):
    """
    Computes the a posteriori probability p(wi|x) for each class wi by dividing
    the number of votes of each label among the k nearest neighbors by K.
    """
    # Compute label votes for k nearest neighbors.
    knn_label_votes = self.knn_label_votes(x)

    # p(wi|x) = num_votes(wi)/K. Map label index into probability.
    return np.array(list(map(
      lambda label: knn_label_votes.get(label, 0) / float(self.K),
      range(self.num_classes),
    )))

  def knn_label_votes(self, x):
    """
    Finds the k nearest neighbors, and counts their labels. Returns a dict
    mapping each label to their count.
    """
    # Evaluate the distance L2 of x to all training points.
    dist  = np.linalg.norm(x - self.x_train, axis=1)
    
    # Compute the indices of the k nearest points (with respect to x_train).
    # Use negative distances to force min-heap behave like a max-heap.
    nearest_k_indices = []
    for i in range(len(dist)):
      heapq.heappush(nearest_k_indices, (-dist[i], i))
      if len(nearest_k_indices) > self.K: heapq.heappop(nearest_k_indices)

    # Count number of votes for each label.
    label_votes = {}
    for label in [self.w_train[k] for (_, k) in nearest_k_indices]:
      label_votes[label] = label_votes.get(label, 0) + 1
    return label_votes

  def evaluate(self, x_test, w_test):
    """
    Returns a tuple (num_correct_predictions, accuracy), meaning the number of
    correct estimations/predictions and the accuracy.
    """
    w_est = self.predict(x_test)
    num_correct_predictions = np.sum(w_est == np.array(w_test))
    accuracy = num_correct_predictions/float(len(w_est))
    return (num_correct_predictions, accuracy)
