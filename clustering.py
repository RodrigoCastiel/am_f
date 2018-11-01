"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs

from core.data_loader import DataLoader
from classifiers.k_means_clustering import KMeansClustering
from classifiers.kcm_f_gh_clustering import KCM_F_GH_Clustering

def main():
  # Set seed for deterministic execution.
  np.random.seed(0)

  # Generate random dataset.
  K = 2
  n_samples = 100
  # X_data, _ = make_blobs(n_samples, centers=K, cluster_std=0.60, random_state=0)
  X_data, _ = datasets.make_moons(n_samples=n_samples, noise=.05)

  # Run KCM-F-GH.
  kcm_f_gh = KCM_F_GH_Clustering(c = K).fit(X_data)
  kcm_f_gh_assignments = kcm_f_gh.get_assigments()

  # Run K-means.
  k_means_clustering = KMeansClustering(K).fit(X_data)
  k_means_assignments = k_means_clustering.get_assigments()

  # Plot dataset and computed means.
  possible_colors = ['r', 'g', 'b', 'm', 'y']
  col_k_means = list(map(lambda k: possible_colors[k], k_means_assignments))
  col_kcm_f_gh = list(map(lambda k: possible_colors[k], kcm_f_gh_assignments))
  
  # Compare results.
  plt.subplot(1, 2, 1)
  plt.scatter(X_data[:, 0], X_data[:, 1], marker='+', c=col_k_means)
  plt.subplot(1, 2, 2)
  plt.scatter(X_data[:, 0], X_data[:, 1], marker='+', c=col_kcm_f_gh)
  plt.show()


if __name__ == "__main__":
    main()
