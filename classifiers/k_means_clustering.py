"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np

epsilon = 1e-6
max_iter = 100

class KMeansClustering:
  """
  Implements LLoyd's Algorithm for clustering a dataset into K groups.
  Initializes the K-means by using Forgy's method.
  """

  def __init__(self, K = 2):
    # Hyper-parameter K: number of clusters.
    self.K = K
    # List of K clusters. Each cluster is specified by a list of point indices.
    self.clusters = []
    # [K x d] matrix whose rows are the cluster means.
    self.k_means = []
    # N-array containing the indices of the assigned means for each point used
    # in the clustering.
    self.assignments = []
    # If true, it will log clustering events.
    self.verbose = True

  def get_cluster_means(self):
    """
    Returns a [K x d] matrix whose rows are the d-dimensional cluster means.
    If the clustering hasn't been performed, returns [].
    """
    return self.k_means

  def get_assigments(self):
    """
    Returns an N-array containing the indices of the assigned means for the
    dataset used in the clustering. If the clustering hasn't been performed, 
    returns [].
    """
    return self.assignments

  def fit(self, x_train):
    """
    Groups x_train into K different clusters. x_train is a [N x d] matrix, where
    N is the number of samples and d is their dimensionality. At the end,
      a. self.clusters will store the groups of points by their indices.
      b. self.k_means will store the k-means coordinates.
      c. self.assigments will store which mean has been assigned to each point.
    """
    # Initialize means (Forgy's method - pick K random points).
    N = len(x_train)
    self.k_means = x_train[np.random.choice(N, self.K), :]
    self.clusters = []

    self.log("+ K-means (k = %d, #training_points = %d)\n" % (self.K, N))
    self.log("Start. Iteration:")

    # Iterative update step.
    for i in range(max_iter):
      self.log(" %d" % (i))
      # Assign nearest mean index for each point in x_train.
      self.assignments = self.assign(x_train)
      # Gather points assigned to the same mean.
      self.clusters = list(map(
        lambda mean_k: [i for i in range(N) if self.assignments[i] == mean_k],
        range(self.K),
      ))
      # Recompute k-means.
      k_means = np.array(list(map(
        lambda cluster: np.mean(x_train[cluster, :], axis=0),
        self.clusters,
      )))
      # Stop condition.
      if np.linalg.norm(k_means - self.k_means) < epsilon:
        break
      self.k_means = k_means

    self.log(". Finish.\n")
    return self

  def assign(self, x_set):
    """
    Assigns a mean to each point in x_set. Uses the current self.k_means member.
    Returns an N-array containing the indices of the assigned means.
    """
    def nearest_mean(x, k_means):
      # Evaluate the distance L2 of x to all means, return index of the nearest.
      dist = np.linalg.norm(x - k_means, axis=1)
      return np.argmin(dist)

    return np.array(list(map(
        lambda x: nearest_mean(x, self.k_means),
        x_set,
      )))

  def log(self, text, **kwargs):
    if self.verbose:
      print(text, end='', flush=True)
