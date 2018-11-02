"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
import random
import sklearn.utils

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from core.data_loader import DataLoader
from classifiers.gaussian_mle import GaussianMLE
from classifiers.knn_classifier import KNNClassifier
from classifiers.combined_max_classifier import CombinedMaxClassifier


num_folds_cv = 10
num_times_cv = 30


def main():
  print("+--------------------------------------+")
  print("|   Machine Learning Project, Part 2   |")
  print("+--------------------------------------+")
  print("|       Author: Rodrigo Castiel        |")
  print("+--------------------------------------+")

  # Set seed for deterministic execution.
  random.seed(0)
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

  # Evaluate estimators.
  evaluate_cross_validation(classifiers, x_train, w_train)
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
  K_optimal = 7 #grid_search.best_params_['K']

  # Build classifier with optimal K.
  return CombinedMaxClassifier(K_optimal, views).fit(x_train, w_train)


def perform_cross_validation(classifier, x_train, w_train):
  """
  Runs cross validation *num_times_cv* times on *classifier*. Returns the
  average accuracy, and its error margin for a 95%-confidence interval.
  Reference:
  http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
  """
  accuracy_cv = []
  for i in range(num_times_cv):
    x_test, y_test = sklearn.utils.shuffle(x_train, w_train, random_state=i)
    skf = StratifiedKFold(n_splits=num_folds_cv)
    accuracy_cv.extend(list(cross_val_score(
      classifier, x_test, y_test, scoring='accuracy', cv=skf, n_jobs=-1
    )))

  avg_accuracy = np.mean(accuracy_cv)
  error_margin = 2*np.std(accuracy_cv)

  return (avg_accuracy, error_margin)

  # return (avg, error_margin)

def evaluate_cross_validation(classifiers, x_train, w_train):
  print(
    "\n------------------- %02dx %02d-fold Cross-Validation ---------------------"
    % (num_times_cv, num_folds_cv)
  )
  print("Classifier" + " "*34 + "Accuracy")

  max_len = 40
  for classifier in classifiers:
    # Perform cross-validation on training set.
    accuracy, margin = perform_cross_validation(classifier, x_train, w_train)

    # Print out their results.
    classifier_name = classifier.get_name()
    num_dots = max_len - len(classifier_name)
    print(
      "+ %s %s %lf%% (+/-%lf%%)"
      %(classifier_name, num_dots*".", 100.0*accuracy, 100.0*margin)
    )

  print("-"*70 + "\n")

def evaluate_accuracy_on_test_set(data_loader, classifiers):
  # Evaluate estimators on test set.
  x_test, w_test = data_loader.test_data()
  data_size = len(w_test)

  print("\n------------------ Accuracy Evaluation on Test Set -------------------")
  print("Classifier" + " "*38 + "Accuracy")

  max_len = 44
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

  print("-"*70 + "\n")


if __name__ == "__main__":
    main()
