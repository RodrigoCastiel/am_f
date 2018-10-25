"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np

from data_loader import DataLoader
from gaussian_mle import GaussianMLE
from knn_classifier import KNNClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from committee_classifier_base import CommitteeClassifierBase

class CombinedMaxClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, K, view_1, view_2, view_3):
    # Committee of classifiers.
    self.gaussian_mle_1 = GaussianMLE()
    self.gaussian_mle_2 = GaussianMLE()
    self.gaussian_mle_3 = GaussianMLE()
    self.knn_classifier_1 = KNNClassifier(K)
    self.knn_classifier_2 = KNNClassifier(K)
    self.knn_classifier_3 = KNNClassifier(K)

    # A view is a subset of features, defined by a list of indices. Here, we are
    # testing GaussianMLE and KNN over different views (specified by views).
    self.view_1 = view_1
    self.view_2 = view_2
    self.view_3 = view_3
    self.L = 3

  def fit(self, x_train, w_train):
    # Train all committee fclassifiers with their corresponding views.
    self.gaussian_mle_1.fit(x_train[:, self.view_1], w_train)
    self.gaussian_mle_2.fit(x_train[:, self.view_2], w_train)
    self.gaussian_mle_3.fit(x_train[:, self.view_3], w_train)
    self.knn_classifier_1.fit(x_train[:, self.view_1], w_train)
    self.knn_classifier_2.fit(x_train[:, self.view_2], w_train)
    self.knn_classifier_3.fit(x_train[:, self.view_3], w_train)

    return self

  def predict(self, x_set):
    def classify(x):
      L = self.L
      p_w = self.gaussian_mle_1.compute_a_priori()
      blended = (1-L) * p_w + self.L * np.max([
          self.gaussian_mle_1.compute_a_posteriori(x[self.view_1]),
          self.gaussian_mle_2.compute_a_posteriori(x[self.view_2]),
          self.gaussian_mle_3.compute_a_posteriori(x[self.view_3]),
          self.knn_classifier_1.compute_a_posteriori(x[self.view_1]),
          self.knn_classifier_2.compute_a_posteriori(x[self.view_2]),
          self.knn_classifier_3.compute_a_posteriori(x[self.view_3]),
        ], axis=0)
      return np.argmax(blended)
    
    return np.array(list(map(classify, x_set)))

  def evaluate(self, x_test, w_test):
    """
    Returns a tuple (num_correct_predictions, accuracy), meaning the number of
    correct estimations/predictions and the accuracy.
    """
    w_est = self.predict(x_test)
    num_correct_predictions = np.sum(w_est == np.array(w_test))
    accuracy = num_correct_predictions/float(len(w_est))
    return (num_correct_predictions, accuracy)
