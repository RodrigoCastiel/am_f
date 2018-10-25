"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from data_loader import DataLoader
from gaussian_mle import GaussianMLE
from knn_classifier import KNNClassifier
from combined_max_classifier import CombinedMaxClassifier

def main():
  # Load training and test data.
  loader = DataLoader()
  loader.load("data/segmentation")
  x_train, w_train  = loader.training_data()

  # Construct and train estimators.
  gaussian_mle = GaussianMLE()
  gaussian_mle.fit(x_train, w_train)

  knn_classifier = KNNClassifier(K = 1)
  knn_classifier.fit(x_train, w_train)

  view_1 = list(range(0, 9))
  view_2 = list(range(9, 19))
  view_3 = list(range(19))
  K = 5
  combined_max_classifier = CombinedMaxClassifier(K, view_1, view_2, view_3)
  combined_max_classifier.fit(x_train, w_train)

  # Evaluate estimators on test set.
  x_test, w_test = loader.test_data()
  num_hits_g, accuracy_g = gaussian_mle.evaluate(x_test, w_test)
  num_hits_k, accuracy_k = knn_classifier.evaluate(x_test, w_test)
  num_hits_c, accuracy_c = combined_max_classifier.evaluate(x_test, w_test)
  data_size = len(w_test)

  # Output data.
  print("Classifier         Accuracy")
  print(
    "Gaussian MLE ..... %lf%% (%d/%d)"
    %(100.0*accuracy_g, num_hits_g, data_size)
  )
  print(
    "KNN .............. %lf%% (%d/%d)"
    %(100.0*accuracy_k, num_hits_k, data_size)
  )
  print(
    "Combined Max ..... %lf%% (%d/%d)"
    %(100.0*accuracy_c, num_hits_c, data_size)
  )

if __name__ == "__main__":
    main()
