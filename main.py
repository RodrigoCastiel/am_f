"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from core.data_loader import DataLoader
from classifiers.gaussian_mle import GaussianMLE
from classifiers.knn_classifier import KNNClassifier
from classifiers.combined_max_classifier import CombinedMaxClassifier

from sklearn.model_selection import GridSearchCV

num_folds_cv = 10

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
  classifiers = [gaussian_mle, knn_classifier, combined_max_classifier]

  # Evaluate overall accuracy of estimators on test data.
  evaluate_accuracy_on_test_set(loader, classifiers)

def train_gaussian_mle(x_train, w_train):
  """
  Constructs and trains a Gaussian Maximum Likelihood Estimator.
  """
  return GaussianMLE().fit(x_train, w_train)


def train_knn_classifier(x_train, w_train):
  """
  Constructs, finds the optimal hyper-parameter K via grid-search, and returns
  a trained KNN Classifier.
  """
  # Perform grid-search to find the optimal K value.
  K_values = [1, 3, 5, 7, 9, 13, 15, 17, 19]
  grid_search = GridSearchCV(KNNClassifier(), {'K': K_values}, n_jobs=-1)
  grid_search.fit(x_train, w_train)
  K_optimal = grid_search.best_params_['K']

  # Build classifier with optimal K.
  return KNNClassifier(K_optimal).fit(x_train, w_train)


def train_combined_classifier(x_train, w_train):
  """
  Constructs, finds the optimal hyper-parameter K via grid-search, and returns
  a trained combined-max classifier.
  """
  # Perform grid-search to find the optimal K value.
  views = [list(range(0, 9)), list(range(9, 19)), list(range(19))]
  K_values = [1, 3, 5, 7, 9, 13, 15, 17, 19]
  grid_search = GridSearchCV(
    CombinedMaxClassifier(views=views),
    param_grid = {'K': K_values},
    n_jobs=-1,
    scoring='accuracy',
    cv=num_folds_cv,
  )
  grid_search.fit(x_train, w_train)
  K_optimal = grid_search.best_params_['K']

  # Build classifier with optimal K.
  return CombinedMaxClassifier(K_optimal, views).fit(x_train, w_train)


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
