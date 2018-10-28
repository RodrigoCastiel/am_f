"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from core.data_loader import DataLoader
from classifiers.gaussian_mle import GaussianMLE
from classifiers.knn_classifier import KNNClassifier
from classifiers.combined_max_classifier import CombinedMaxClassifier

def main():
  # Set seed for deterministic execution.
  np.random.seed(0)

  # Load training and test data.
  loader = DataLoader()
  loader.load("data/segmentation")
  x_train, w_train  = loader.training_data()

  # Build and train classifiers with the optimal hyper-parameters.
  gaussian_mle = train_gaussian_mle(x_train, w_train)
  knn_classifier = train_knn_classifier(x_train, w_train)
  combined_max_classifier = train_combined_classifier(x_train, w_train)

  # Evaluate overall accuracy of estimators on test data.
  classifiers = [
    gaussian_mle,
    knn_classifier,
    combined_max_classifier,
  ]
  evaluate_accuracy_on_test_set(loader, classifiers)

def train_gaussian_mle(x_train, w_train):
  """
  Constructs and trains a Gaussian Maximum Likelihood Estimator.
  """
  return GaussianMLE().fit(x_train, w_train)

def train_knn_classifier(x_train, w_train):
  """
  Constructs and trains a KNN Classifier.
  """
  knn_classifier = KNNClassifier(K = 1)
  knn_classifier.fit(x_train, w_train)
  return knn_classifier

def train_combined_classifier(x_train, w_train):
  """
  Construct and train the combined model classifier.
  """
  K = 5
  views = [[0], list(range(0, 9)), list(range(9, 19)), list(range(19))]
  combined_max_classifier = CombinedMaxClassifier(K, views)
  combined_max_classifier.fit(x_train, w_train)
  return combined_max_classifier

def evaluate_accuracy_on_test_set(data_loader, classifiers):
  # Evaluate estimators on test set.
  x_test, w_test = data_loader.test_data()
  data_size = len(w_test)

  print("\n---------------- Accuracy Evaluation on Test Set -----------------")
  print("Classifier" + " "*34 + "Accuracy")

  max_len = 40
  for classifier in classifiers:
    # Evaluate the accuracy of each classifier on (x_test, w_test).
    num_hits, accuracy = classifier.evaluate(x_test, w_test)

    # Print out their results.
    classifier_name = classifier.get_name()
    num_dots = max_len - len(classifier_name)
    print(
      "+ %s %s %lf%% (%d/%d)"
      %(classifier_name, num_dots*".", 100.0*accuracy, num_hits, data_size)
    )

  print("------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
