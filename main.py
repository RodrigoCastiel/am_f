"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from data_loader import DataLoader
from gaussian_mle import GaussianMLE
from knn_classifier import KNNClassifier

def main():
  # Load training and test data.
  loader = DataLoader()
  loader.load("data/segmentation")
  x_train, w_train  = loader.training_data()

  # Construct and train estimators.
  gaussian_mle = GaussianMLE()
  gaussian_mle.train(x_train, w_train)
  knn_classifier = KNNClassifier()
  knn_classifier.train(x_train, w_train)

  # Evaluate estimators on test set.
  x_test, w_test = loader.test_data()
  correct_predictions_g, accuracy_g = gaussian_mle.evaluate(x_test, w_test)
  correct_predictions_k, accuracy_k = knn_classifier.evaluate(x_test, w_test)
  data_size = len(w_test)

  # Output data.
  print("Classifier         Accuracy")
  print(
    "Gaussian MLE ..... %lf%% (%d/%d)"
    %(100.0*accuracy_g, correct_predictions_g, data_size)
  )
  print(
    "KNN .............. %lf%% (%d/%d)"
    %(100.0*accuracy_k, correct_predictions_k, data_size)
  )

if __name__ == "__main__":
    main()
