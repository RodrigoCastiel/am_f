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

  # Construct and train maximum likelihood gaussian estimator.
  gaussian_mle = GaussianMLE()
  gaussian_mle.train(x_train, w_train)

  # Evaluate GaussianMLE on test set.
  x_test, w_test = loader.test_data()
  num_correct_predictions, accuracy = gaussian_mle.evaluate(x_test, w_test)
  print(
    "Accuracy ... %lf%% (%d/%d)"
    %(100.0*accuracy, num_correct_predictions, len(w_test))
  )

if __name__ == "__main__":
    main()
