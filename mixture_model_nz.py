import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import norm, gamma

model_dict = {
              #'gaussian' : norm.pdf,
              'gamma' : gamma
              }

class mixture_nz:

  # class for a gmm nz with a vector of gaussian/gamma parameters
  # weights, means, variances

  def __init__(self, weights, pars, locs, scales, models='gamma'):
    self.n_components = np.size(weights)
    self.pars = np.atleast_1d(pars)
    self.weights = np.atleast_1d(weights)
    self.locs = np.atleast_1d(locs)
    self.scales = np.atleast_1d(scales)
    
    if np.size(models) > 1:
      self.models = np.asarray(models)
    else:
      self.models = np.asarray([models]*self.n_components)


  def bulk_mean_for_mixture(m_mix, w_b, a_b, theta_b, w_o, m_o, a_o, theta_o):

    retVar = m_mix/w_b - (w_o/w_b)*(a_o*theta_o + m_o) - a_b*theta_b

    return retVar

  def get_mean(self):

    dist_mean = 0.

    for im,model in enumerate(self.models):
      dist_mean += self.weights[im]*model_dict[model](self.pars[im], self.locs[im], self.scales[im]).mean()

    return dist_mean


  def gridded_nz(self, zmin, zmax, npix=512, normed=False):

    nz = np.zeros(npix)
    z = np.linspace(zmin, zmax, npix)

    for im,model in enumerate(self.models):
      nz += self.weights[im]*model_dict[model].pdf(z, self.pars[im], self.locs[im], self.scales[im])

    if normed:
      nz = nz / cumtrapz(nz, z)[-1]

    return nz


if __name__=='__main__':

  from matplotlib import pyplot as plt
  from matplotlib import rc

  rc('text', usetex=True)
  rc('font', family='serif')
  rc('font', size=11)

  plt.close('all')

  zmin = 0.0
  zmax = 5.0
  npix = 512
  z = np.linspace(zmin, zmax, npix)

  bulk = mixture_nz(1., 3., 0.0225, 0.25)
  outliers = mixture_nz(1, 3., 0.1, 0.05)
  bulk_outliers = mixture_nz([1.,0.1], [3., 2.5], [0., 0.1], [0.25, 0.05])

  print(bulk.get_mean())
  print(outliers.get_mean())
  print(bulk_outliers.get_mean())

  plt.figure(1, figsize=(4.5,3.75))
  plt.plot(z, bulk.gridded_nz(zmin, zmax, npix, True), label='Bulk')
  plt.plot(z, 0.1*outliers.gridded_nz(zmin, zmax, npix, True), label='Outliers')
  plt.plot(z, bulk_outliers.gridded_nz(zmin, zmax, npix, True), label='Bulk$+ 0.1$Outliers')
  plt.legend()
  plt.xlabel('Redshift $z$')
  plt.ylabel('$n(z)$')
  plt.savefig('./plots/bulk_outliers.png', dpi=300, bbox_inches='tight')
