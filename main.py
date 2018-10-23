"""
Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

import numpy as np
from data_loader import DataLoader
from gaussian_mle import GaussianMLE

def main():
  # Load training and test data.
  loader = DataLoader()
  loader.load("data/segmentation")
  x_train, w_train  = loader.training_data()
  num_classes = loader.get_num_classes()

  # Construct and train maximum likelihood gaussian estimator.
  gaussian_mle = GaussianMLE()
  gaussian_mle.train(x_train, w_train, num_classes)
  print(w_train)
  print(gaussian_mle.get_a_priori_probabilities())

if __name__ == "__main__":
    main()
