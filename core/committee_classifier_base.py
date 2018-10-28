"""
This module defines the base class for bayesian probability based committee
classifiers. Such classifiers are combined as a committee through the max-rule,
defined in http://www.cin.ufpe.br/~fatc/AM/Projeto-AM-2018-2.pdf.

Author: Rodrigo Castiel, Federal University of Pernambuco (UFPE).
"""

from sklearn.base import BaseEstimator, ClassifierMixin

class CommitteeClassifierBase(BaseEstimator, ClassifierMixin):
  def compute_a_posteriori(self, x):
    """
    Must return a list containing the posteriori probabilities p(wi|x) for each 
    class wi.
    """
    raise NotImplementedError("Implement to return [p(w0|x), p(w1|x), ...]")
