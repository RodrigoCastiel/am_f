"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
import random
import sklearn.utils
from sklearn import metrics

from core.data_loader import DataLoader
from classifiers.kcm_f_gh_clustering import KCM_F_GH_Clustering
from classifiers.k_means_clustering import KMeansClustering


def main():
  print("+--------------------------------------+")
  print("|   Machine Learning Project, Part 1   |")
  print("+--------------------------------------+")
  print("|       Author: Rodrigo Castiel        |")
  print("+--------------------------------------+")
  print()

  # Set seed for deterministic execution.
  random.seed(0)
  np.random.seed(0)

  # Load training and test data.
  loader = DataLoader()
  loader.load("data/segmentation")
  x_data, w_data  = loader.test_data()

  K = len(np.unique(w_data))
  shape_view = [0, 1, 3, 5, 6, 7, 8]
  rgb_view = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

  ## SHAPE VIEW ----------------------------------------------------------------
  print("+--------------------+")
  print("|     SHAPE VIEW     |")
  print("+--------------------+")

  # Run KCM-F-GH.
  kcm_shape_view = KCM_F_GH_Clustering(c = K).fit(x_data[:, shape_view], w_data)
  kcm_shape_view_assignments = kcm_shape_view.get_assigments()
  rand_score = metrics.adjusted_rand_score(w_data, kcm_shape_view_assignments)
  print("> Adjusted rand score: ", rand_score)
  print()
  
  # Run k-means.
  k_means_shape = KMeansClustering(K).fit(x_data[:, shape_view])
  k_means_shape_assignments = k_means_shape.get_assigments()
  rand_score = metrics.adjusted_rand_score(w_data, k_means_shape_assignments)
  print("> Adjusted rand score: ", rand_score)
  print()

  ## RGB VIEW ------------------------------------------------------------------
  print("+--------------------+")
  print("|      RGB VIEW      |")
  print("+--------------------+")

  # Run KCM-F-GH.
  kcm_rgb_view = KCM_F_GH_Clustering(c = K).fit(x_data[:, rgb_view], w_data)
  kcm_rgb_view_assignments = kcm_rgb_view.get_assigments()
  rand_score = metrics.adjusted_rand_score(w_data, kcm_rgb_view_assignments)
  print("> Adjusted rand score: ", rand_score)
  print()

  # Run k-means.
  k_means_rgb = KMeansClustering(K).fit(x_data[:, shape_view])
  k_means_rgb_assignments = k_means_rgb.get_assigments()
  rand_score = metrics.adjusted_rand_score(w_data, k_means_rgb_assignments)
  print("> Adjusted rand score: ", rand_score)
  print()


if __name__ == "__main__":
    main()
