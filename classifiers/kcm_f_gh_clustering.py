"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import itertools
import numpy as np
import random

epsilon = 1e-6
max_iter = 100

class KCM_F_GH_Clustering:
  """
  Implements KCM-F-GH algorithm proposed in 'Gaussian kernel c-means hard 
  clustering algorithms with automated computation of the width 
  hyper-parameters'. Reference:
  https://www.sciencedirect.com/science/article/pii/S0031320318300712

  This is a hard clustering algorithm in feature-space, where the gaussian
  kernel parameters (i.e., delta) are automatically computed.
  """

  def __init__(self, c = 2):
    # Hyper-parameter c: number of clusters.
    self.c = c
    # List of c clusters. Each cluster is specified by a list of point indices.
    self.clusters = []
    # N-array containing the indices of the assigned means for each point used
    # in the clustering.
    self.assignments = []
    # Squared inverse of width hyper-parameter.
    self.inv_s2 = []
    # Training points used in clustering.
    self.x_set = []

  def get_assigments(self):
    """
    Returns an N-array containing the indices of the assigned means for the
    dataset used in the clustering. If the clustering hasn't been performed, 
    returns [].
    """
    return self.assignments

  def fit(self, x_train):
    """
    Groups x_train into c different clusters. x_train is a [N x p] matrix, where
    N is the number of samples and p is their dimensionality. At the end,
      a. self.clusters will store the groups of points by their indices.
      b. self.assigments will store which mean has been assigned to each point.
    """
    N = len(x_train)
    # Estimate initial value of (1/s^2).
    self.inv_s2 = KCM_F_GH_Clustering.estimate_initial_s_parameter(x_train)
    # Initialize c clusters, each with a randomly picked point (representative).
    self.clusters = list(map(lambda i: [i], random.sample(range(N), self.c)))
    # Assign clusters for each point in x_train (with current self.clusters).
    self.assignments = self.assign(x_train)
    # Update clusters to add the remaining points.
    self.clusters = KCM_F_GH_Clustering.build_clusters(self.assignments, self.c)

    # Iterative update step.
    for _ in range(max_iter):
      break

    return self

  def assign(self, x_set):
    """
    Assigns each point xk in *x_set* to a cluster i, where i = 0..(c-1).
    It computes the euclidian distance between xk and the representatives of all
    clusters in feature space. That is, || phi(xk) - gi ||^2. Then, it chooses
    the nearest cluster i.
    """
    def dist_to_cluster(i):
      """
      Returns the distance of each point xk in x_set to the cluster i. That is,
        || phi(xk) - gi ||^2, equation (21).
      The returned array is [N x 1], where N is the number of points in x_set.
      """
      return np.zeros((len(x_set)))

    # Compute distances in feature space, a [c x N] matrix. Each row i means the
    # distance between each point xj in x_set to cluster i.
    distances = np.array(list(map(dist_to_cluster, range(self.c))))

    # Assign cluster with minimal distance to each point j. For each column 
    # (point j), we pick the row with the smallest distance.
    return np.argmin(distances, axis=1)

  def kernel(self, xl, xk):
    """
    Returns the gaussian kernel with hyper-parameter s evaluated for xl and xk.
    Equation (9).
    """
    return np.exp(-0.5 * np.sum((xl - xk)**2 * self.inv_s2))

  @staticmethod
  def estimate_initial_s_parameter(x_train):
    """
    Estimates the initial value for the global width hyper-parameter s given the
    input dataset *x_train*.
    """
    def dist(i, j):
      return np.linalg.norm(x_train[i] - x_train[j])

    # Compute distances between all points, then find 10% and 90% percentiles.
    idx = range(len(x_train))
    dists = [dist(i, j) for (i, j) in itertools.product(idx, idx) if i < j]
    p10, p90 = np.percentile(dists, [10., 90.])
    avg = 0.5 * (p10 + p90)

    # Return 1/s2.
    return np.full((x_train.shape[1]), 1.0/avg)

  @staticmethod
  def build_clusters(assignments, c):
    """
    Given an array of N assignments [c1, c2, ..., ck], where cj is the cluster
    index for a point at index i = 0..(N-1), it returns a list of c clusters
    containing the indices i that share the same cluster. 

    Example:
    Let c = 3 and assignments = [1, 2, 0, 2, 2, 1, 1]. Then it would return
    [[2], [0, 5, 6], [1, 4, 5]].
    """
    N = len(assignments)
    return list(map(
      lambda cluster: [i for i in range(N) if assignments[i] == cluster],
      range(c),
    ))
