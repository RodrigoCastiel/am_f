"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from core.data_loader import DataLoader
from classifiers.gaussian_mle import GaussianMLE
from classifiers.knn_classifier import KNNClassifier
from classifiers.combined_max_classifier import CombinedMaxClassifier

def evaluate_accuracy_on_test_set(
  data_loader, 
  gaussian_mle,
  knn_classifier, 
  combined_model
):
  # Evaluate estimators on test set.
  x_test, w_test = data_loader.test_data()
  num_hits_g, accuracy_g = gaussian_mle.evaluate(x_test, w_test)
  num_hits_k, accuracy_k = knn_classifier.evaluate(x_test, w_test)
  num_hits_c, accuracy_c = combined_model.evaluate(x_test, w_test)
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

def main():
  # Set seed for deterministic execution.
  np.random.seed(0)

  # Load training and test data.
  loader = DataLoader()
  loader.load("data/segmentation")
  x_train, w_train  = loader.training_data()

  # Construct and train gaussian maximum likelihood estimator.
  gaussian_mle = GaussianMLE()
  gaussian_mle.fit(x_train, w_train)

  # Construct and train KNN classifier.
  knn_classifier = KNNClassifier(K = 1)
  knn_classifier.fit(x_train, w_train)

  # Construct and train the combined model classifier.
  K = 5
  views = [list(range(0, 9)), list(range(9, 19)), list(range(19))]
  combined_max_classifier = CombinedMaxClassifier(K, views)
  combined_max_classifier.fit(x_train, w_train)

  # Evaluate overall accuracy of estimators on test data.
  evaluate_accuracy_on_test_set(
    loader,
    gaussian_mle,
    knn_classifier,
    combined_max_classifier,
  )

if __name__ == "__main__":
    main()
