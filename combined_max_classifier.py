"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
import itertools

from data_loader import DataLoader
from gaussian_mle import GaussianMLE
from knn_classifier import KNNClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from committee_classifier_base import CommitteeClassifierBase

class CombinedMaxClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, K, views):
    def classifier(name):
      return GaussianMLE() if name == "gaussian_mle" else KNNClassifier(K)

    # Cartesian product between list of classifiers and list of views.
    classifiers = ["gaussian_mle", "knn_classifier"]
    prod = itertools.product(classifiers, views)

    # Map classifier names into actual classifier objects.
    self.committee = [(classifier(name), view) for (name, view) in prod]

    # A view is a subset of features, defined by a list of indices. Here, we are
    # testing GaussianMLE and KNN over different views (specified by views).
    self.L = len(views)

  def fit(self, x_train, w_train):
    # Train all committee classifiers with their corresponding views.
    for (classifier, view) in self.committee:
      classifier.fit(x_train[:, view], w_train)

    # Estimate a prior probabilities p(wi) for each class wi.
    self.p_w = DataLoader.compute_a_priori(w_train)

    return self

  def predict(self, x_set):
    def classify(x):
      L = self.L
      p_w = self.p_w
      p_w_x_classifiers = [classifier.compute_a_posteriori(x[view])
                            for (classifier, view)
                            in self.committee]
      blended = (1-L) * p_w + self.L * np.max(p_w_x_classifiers, axis=0)
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
