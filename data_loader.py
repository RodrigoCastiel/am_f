"""
This module defines the class DataLoader, in charge of reading the Image Segmentation Data from:
  http://archive.ics.uci.edu/ml/machine-learning-databases/image/
Additionally, it has helper methods to automatically generate cross-validation datasets for hyper-
parameters tuning.

Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import csv
import numpy as np
import random

class DataLoader:
  """
  Usage.
    a. Loading:
      import data_loader
      loader = data_loader.DataLoader()
      loader.load("data/segmentation")
      print(loader.get_labels())
      print(loader.get_features())
    b. Retrieving data:
      loader.training_data()
      loader.test_data()
    c. Cross validation sets:
      loader.generate_cross_validation_sets(number_folds)
  """

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

  def test_data(self):
    """
    Returns the loaded test data (x_test, w_test).
      x_test is the list of test samples [x1, x2, ..., xn].
      w_test is the list of integer labels [w0, w1, ..., wn].
    """
    return (self.x_test, self.w_test)

  def get_num_classes(self):
    """Returns the number of loaded classes/labels."""
    return len(self.labels)

  def get_labels(self):
    """
    Returns the list of loaded classes/labels.
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

  def generate_cross_validation_sets(self, num_folds):
    """
    Splits up the training data into *num_folds* random, disjoint sets for cross-validation.
    Returns a list of pairs containing the smaller sub-sets and their labels.
    E.g., [(x_train1, w_train1), (x_train2, w_train2), ...].
    """
    def chunk_to_subset(chunk):
      return (
        list(map(lambda i: self.x_train[i], chunk)),
        list(map(lambda i: self.w_train[i], chunk))
      )

    def break_down_chunks(seq, num_folds):
        avg = len(seq) / float(num_folds)
        chunks = []
        last = 0.0
        while last < len(seq):
            chunks.append(seq[int(last):int(last + avg)])
            last += avg
        return chunks

    shuffled_indices = list(range(len(self.x_train)))
    random.shuffle(shuffled_indices)

    index_chunks = break_down_chunks(shuffled_indices, num_folds)
    return list(map(chunk_to_subset, index_chunks))

  @staticmethod
  def group_by_label(x_data, w_data, int_labels):
    """
    Given a dataset represented by (x_data, w_data), returns a list of subsets
    grouped by their integer label specified by w_data. The out list contains
    only features, and follows the label order of int_labels.
      E.g., [x_data1, x_data2, ..., x_dataN], where N is the number of labels.
    """
    return list(map(
      lambda label_k: [x_data[i] for i in range(len(w_data)) if w_data[i] == label_k],
      int_labels
    ))

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
