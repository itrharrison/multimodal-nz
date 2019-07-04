import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import norm, gamma

model_dict = {'gaussian' : norm.pdf,
              'gamma' : gamma.pdf}

class mixture_nz:

  # class for a gmm nz with a vector of gaussian/gamma parameters
  # weights, means, variances

  def __init__(self, weights, means, variances, models='gaussian'):
    self.n_components = np.size(weights)
    self.weights = np.atleast_1d(weights)
    self.means = np.atleast_1d(means)
    self.variances = np.atleast_1d(variances)
    
    if np.size(models) > 1:
      self.models = np.asarray(models)
    else:
      self.models = np.asarray([models]*self.n_components)

  def mean_z(self):
    mean_z = np.sum(self.weights*self.means)
    return mean_z

  def gridded_nz(self, zmin, zmax, npix=512, normed=False):

    nz = np.zeros(npix)
    z = np.linspace(zmin, zmax, npix)

    for im,model in enumerate(self.models):
      nz += self.weights[im]*model_dict[model](z, self.means[im], self.variances[im])

    if normed:
      nz = nz / cumtrapz(nz, z)[-1]

    return nz


if __name__=='__main__':

  m1 = mixture_nz(1., 1., 0.5)
  m2 = mixture_nz(1., 0.1, 0.5)
  m12 = mixture_nz([1.0, 0.5], [1.0, 0.1], [0.5, 0.5])

  nz1 = m1.gridded_nz(0,2)
  nz2 = m2.gridded_nz(0,2)
  nz12 = m12.gridded_nz(0,2)
