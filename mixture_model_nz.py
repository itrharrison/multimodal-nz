# Module to create toy n(z)s based on multiple component Gamma distributions

import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import norm, gamma

model_dict = {
              #'gaussian' : norm.pdf,
              'gamma' : gamma
              }

class mixture_nz:

  # class for a mixture model nz with a vector of gaussian/gamma parameters
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
    # Function to calculate the location to assign the bulk, such that the mean of the
    # bulk+outlier n(z) has the value m_mix, for a given outlier n(z)
    # The outlier n(z) is specified by its weight w_o, its location m_o,
    # its shape parameter a_o and its scale parameter theta_o
    # The bulk n(z) is specified by its weight w_b, shape parameter a_b and scale theta_b

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

  # Load in the 2point code
  import sys
  #sys.path.append('~/Dropbox/scratch/git/2point/twopoint')
  sys.path.append('../2point/twopoint')
  from twopoint import *

  rc('text', usetex=True)
  rc('font', family='serif')
  rc('font', size=11)

  plt.close('all')

  zmin = 0.0
  zmax = 5.0
  npix = 512
  z = np.linspace(zmin, zmax, npix)

  reference = mixture_nz(1., 3., 0.0225, 0.25)
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

  # Specify 5 n(z)s
  nzbin = 5
  means = [0.2 0.4 0.6 0.8 1.0]
  outlier_fractions = [0.1 0.1 0.1 0.1 0.1]
  outlier_means = [0.1 0.1 0.1 0.1 0.1]
  outlier_shapes = [??]
  outlier_scales = [??]
  bulk_shapes = [??]
  bulk_scales = [??]

  # Now convert these into the mixture_nz inputs..
  izbin = 0
  mean = means[izbin]
  outlier_fraction = outlier_fractions[izbin]
  outlier_mean = outlier_means[izbin]
  w_b = 1 # Fix the arbitrary scaling of the height of the n(z)s arbitrarily
  w_o = w_b * outlier_fraction # check this is what ws actually do!
  m_o = outlier_mean
  a_b = bulk_shapes[izbin]
  theta_b =
  a_o = outlier_shapes[izbin]
  theta_o =

      # Function to calculate the location to assign the bulk, such that the mean of the
    # bulk+outlier n(z) has the value m_mix, for a given outlier n(z)
    # The outlier n(z) is specified by its weight w_o, its location m_o,
    # its shape parameter a_o and its scale parameter theta_o
    # The bulk n(z) is specified by its weight w_b, shape parameter a_b and scale theta_b


  bulk_mean_for_mixture(mean, w_b, a_b, theta_b, w_o, m_o, a_o, theta_o):
  bulk_outliers = mixture_nz([1.,0.1], [3., 2.5], [0., 0.1], [0.25, 0.05])

  ## Save them in shear-2pt format

  # Load in the fits file that came with the 2point code
  #filename = '~/Dropbox/scratch/git/2point/des_multiprobe_v1.10.fits'
  filename = '../2point/des_multiprobe_v1.10.fits'
  T = TwoPointFile.from_fits(filename, covmat_name=None)

  # Create new n(z)s using the mixture model
  # We just want to modify the source n(z)s (not the lens n(z)s)
  # Assume for now this is hard-wired to be the second kernel...!!! not great coding!!!
  kernel = T.kernels[1]
  print(kernel.name)
  print('Panic if the above doesnt say NZ_SOURCE') # Do it better!
  nbin = kernel.nbin
  z = kernel.z
  zmin = z[0]
  nsample = len(z)
  zmax = z[nsample-1]
  for i in range(nbin):
    # Bodge something together for now to illustrate the point
    kernel.nzs[i] = bulk_outliers.gridded_nz(zmin, zmax, nsample, True) + i
  T.kernels[1] = kernel

  # Check and write out the new n(z)s
  output_root = 'plots/mixture_nz_des_multiprobe_20190717_2100'
  T.plots(output_root, plot_cov=False)
  out_fits_filename = 'mixture_nz_des_multiprobe_20190717_2100.fits'
  T.to_fits(filename = out_fits_filename)
