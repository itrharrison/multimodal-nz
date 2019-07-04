import numpy as np

class mixture_nz:

  # class for a gmm nz with a vector of gaussian/gamma parameters
  # weights, means, variances

  def __init__(self, weights, means, variances, models='gaussian'):
    self.n_components = len(weights)
    self.weights = np.asarray(weights)
    self.means = np.asarray(means)
    self.variances = np.asarray(variances)
    if isiterable(models):
      self.models = np.asarray(models)
    else:
      self.models = np.asarray([models]*self.n_components)

    def get_mean_z(self):
      mean_z = np.sum(self.weights*self.means)

      return mean_z


if __name__=='__main__':

  single_mode = mixture_nz(1., 1., 0.5)
  double_mode = mixture_nz([1.0, 0.5], [1.0, 0.1], [0.5, 0.5])