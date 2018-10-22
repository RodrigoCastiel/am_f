"""
This module defines the class DataLoader, in charge of reading the Image Segmentation Data from:
  http://archive.ics.uci.edu/ml/machine-learning-databases/image/
Additionally, it has helper methods to automatically generate cross-validation datasets for hyper-
parameters tuning.
"""

import csv
import numpy as np

class DataLoader:
  def __init__(self):
    self.features, self.labels = [], []
    self.x_train, self.w_train = [], []
    self.x_test, self.w_test = [], []
    self.lookup_labels = {}

  def training_data(self):
    """
    Returns the loaded training data (x_train, w_train).
      x_train is the list of training samples [x1, x2, ..., xn].
      w_train is the list of integer labels [w0, w1, ..., wn].
    """
    return (self.x_train, self.w_train)

  def get_labels(self):
    """
    Returns the list of loaded labels.
      e.g., ["brickface", "sky", "foliage", "cement", "window", "path", "grass"].
    """
    return self.labels

  def get_features(self):
    """
    Returns the list of loaded features.
      e.g., ["REGION-CENTROID-COL","REGION-CENTROID-ROW","REGION-PIXEL-COUNT", ...].
    """
    return self.features

  def load(self, dataset_prefix):
    """
    Loads training data from the CSV files starting with *dataset_prefix*. That is,
      Metadata: suffix = ".metadata.txt". Contains features and label.
      Training: suffix = ".training.txt". Specifies training data.
      Test:     suffix = "test.txt". Specifies test data.
    """
    metadata_filepath = dataset_prefix + ".metadata.txt"
    training_data_filepath = dataset_prefix + ".training.txt"
    test_data_filepath = dataset_prefix + ".test.txt"

    self.features, self.labels = DataLoader.load_metadata(metadata_filepath)
    self.lookup_labels = {self.labels[i]:i for i in range(len(self.labels))}

    self.x_train, self.w_train = DataLoader.load_samples(training_data_filepath, self.lookup_labels)
    self.x_test, self.w_test = DataLoader.load_samples(test_data_filepath, self.lookup_labels)

  @staticmethod
  def load_samples(filepath, lookup_labels):
    x_samples = []
    w_samples = []
    with open(filepath) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in [row for row in csv_reader if row]:
        # Look up label index.
        w_samples.append(lookup_labels[row[0]])
        # Convert list of string numbers into numpy array.
        x_samples.append(np.array([float(x) for x in row[1:]]))

    return (x_samples, w_samples)

  @staticmethod
  def load_metadata(filepath):
    with open(filepath) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      rows = [row for row in csv_reader if row]
      features = rows[0]
      labels = rows[1]
      return (features, labels)
