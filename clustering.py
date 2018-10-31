"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import DataLoader
from classifiers.k_means_clustering import KMeansClustering


def generate_random_data(num_groups, group_size, spread=0.1, d=2):
  X_data = np.zeros((num_groups*group_size, d))
  i = 0
  for _ in range(num_groups):
    mean = np.random.random((d))
    std = spread*np.random.random((1))
    for _ in range(group_size):
      X_data[i, :] = mean + std*np.random.randn(d)
      i += 1
  np.random.shuffle(X_data)
  return X_data


def main():
  # Set seed for deterministic execution.
  np.random.seed(0)

  # Generate random dataset.
  spread, d = 0.5, 2
  K, group_size = 5, 100
  X_data = generate_random_data(K, group_size, spread, d)

  # Run K-means.
  k_means_clustering = KMeansClustering(K)
  k_means_clustering.fit(X_data)
  assignments = k_means_clustering.get_assigments()
  k_means = k_means_clustering.get_cluster_means()

  # Plot dataset and computed means.
  possible_colors = ['r', 'g', 'b', 'm', 'y']
  data_colors = list(map(lambda k: possible_colors[k], assignments))
  mean_colors = list(map(lambda k: possible_colors[k], range(K)))

  plt.scatter(x=X_data[:, 0], y=X_data[:, 1], marker='+', c=data_colors)
  plt.scatter(x=k_means[:, 0], y=k_means[:, 1], marker='o', c=mean_colors)
  plt.show()


if __name__ == "__main__":
    main()
