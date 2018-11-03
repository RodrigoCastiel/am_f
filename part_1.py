"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import argparse
import numpy as np
import random
import sklearn.utils
import sys
from sklearn import metrics

from core.data_loader import DataLoader
from classifiers.kcm_f_gh_clustering import KCM_F_GH_Clustering
from classifiers.k_means_clustering import KMeansClustering

# Script arguments - to define different views.
parser = argparse.ArgumentParser(description="")
parser.add_argument('views', metavar='N', type=str, nargs='+',
                    help='list of views: [FULL, RGB, SHAPE].')
args = parser.parse_args()
print(args)

num_times = 2

def main():
  print("+--------------------------------------+")
  print("|   Machine Learning Project, Part 1   |")
  print("+--------------------------------------+")
  print("|       Author: Rodrigo Castiel        |")
  print("+--------------------------------------+")
  print()

  # Set seed for deterministic execution.
  random.seed(42)
  np.random.seed(42)

  # Load training and test data.
  loader = DataLoader()
  loader.load("data/segmentation")
  x_data, w_data  = loader.training_data()

  K = len(np.unique(w_data))
  shape_view = [0, 1, 3, 5, 6, 7, 8]
  rgb_view = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  full_view = shape_view + rgb_view

  if "FULL" in args.views:
    ## FULL VIEW ---------------------------------------------------------------
    print("+--------------------+")
    print("|      FULL VIEW     |")
    print("+--------------------+")
    evaluate_kcm_f_gh(K, num_times, x_data[:, full_view], w_data)
    evaluate_k_means(K, x_data[:, full_view], w_data)
  if "RGB" in args.views:
    ## RGB VIEW ----------------------------------------------------------------
    print("+--------------------+")
    print("|      RGB VIEW      |")
    print("+--------------------+")
    evaluate_kcm_f_gh(K, num_times, x_data[:, rgb_view], w_data)
    evaluate_k_means(K, x_data[:, rgb_view], w_data)
  if "SHAPE" in args.views:
    ## SHAPE VIEW --------------------------------------------------------------
    print("+--------------------+")
    print("|     SHAPE VIEW     |")
    print("+--------------------+")
    evaluate_kcm_f_gh(K, num_times, x_data[:, shape_view], w_data)
    evaluate_k_means(K, x_data[:, shape_view], w_data)


def evaluate_kcm_f_gh(K, M, x_data, w_data):
  """
  Performs KCM-F-GH clustering M times with c = K on (x_data, w_data).
  Logs the best fit data (by fit_error).
  """
  print("KCM_F_GH (c = %d, #points = %d)" % (K, len(w_data)))
  results = [train_kcm_f_gh(K, i, x_data, w_data) for i in range(M)]
  fit_error, rand_score = min(results)
  print()
  print("> Adjusted rand score: ", rand_score)
  print("> Best fit error: ", fit_error)
  print()


def train_kcm_f_gh(K, i, x_data, w_data):
  print("Run %d. " % (i), end='', flush=True)
  kcm = KCM_F_GH_Clustering(c = K)
  kcm.fit(x_data, w_data)
  kcm_assignments = kcm.get_assignments()
  rand_score = metrics.adjusted_rand_score(w_data, kcm_assignments)
  return kcm.get_fit_error(), rand_score


def evaluate_k_means(K, x_data, w_data):
  """
  Performs K-means clustering  (x_data, w_data).
  Logs the adjusted rand score.
  """
  k_means_full = KMeansClustering(K).fit(x_data)
  k_means_full_assignments = k_means_full.get_assignments()
  rand_score = metrics.adjusted_rand_score(w_data, k_means_full_assignments)
  print("> Adjusted rand score: ", rand_score)
  print()


if __name__ == "__main__":
  main()
