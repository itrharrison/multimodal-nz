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
    self.pars = np.atleast_1d(pars) # This is 'a' in scipiy notation, or  'k' in wikipedia notation
    self.weights = np.atleast_1d(weights)
    self.locs = np.atleast_1d(locs)
    self.scales = np.atleast_1d(scales) # This is 'theta' in wikipedia notation

    if np.size(models) > 1:
      self.models = np.asarray(models)
    else:
      self.models = np.asarray([models]*self.n_components)

  def bulk_mean_for_mixture(mix_mean, b_w, b_a, b_theta, o_w, o_loc, o_a, o_theta):
  #def bulk_mean_for_mixture(mean_mix, w_b, a_b, theta_b, w_o, loc_o, a_o, theta_o):
    # Function to calculate the mean to assign the bulk, such that the mean of the
    # bulk+outlier n(z) has the value m_mix, for a given outlier n(z)
    # The outlier n(z) is specified by its weight w_o, its location loc_o,
    # its shape parameter a_o and its scale parameter theta_o
    # The bulk n(z) is specified by its weight w_b, shape parameter a_b and scale theta_b

    # This was previously calculating the location instead of the mean but I think 
    # the mean is a more useful thing to return so switching to that instead
    #retVar = m_mix/w_b - (w_o/w_b)*(a_o*theta_o + loc_o) - a_b*theta_b

    # Writing this out with more steps for clarity:
    b_gamma_mean = b_a * b_theta
    o_gamma_mean = o_a * o_theta
    o_mean = o_gamma_mean + o_loc
    b_mean = ( mix_mean - o_w * o_mean ) / b_w

    return b_mean

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

def Gamma_skew_to_a(skew):

  # From wikipedia `The skewness of the gamma distribution only depends on its 
  # shape parameter, k, and it is equal to 2/{\sqrt {k}}.''
  # So to convert from skew to shape we do k = (2/skew)^2
  # k is the shape parameter 
  # k is written as 'a' in the scipy documentation so use that notation here
  a = (2.0 / skew) * (2.0 / skew) 

  return a

def Gamma_mean_a_loc_to_theta(mean, a, loc):

  # loc is the scipy notation for an offset of the whole Gamma distribution
  # Now the difference between the mean and the loc determines the variance
  # scipy calls theta the 'scale'
  gamma_mean = mean - loc
  theta = gamma_mean / a

  return theta

def Gamma_a_to_skew(a):

  skew = 2 / np.sqrt(a)

  return skew

def Gamma_a_theta_loc_to_mean(a, theta, loc):

  # First calculate the mean of the Gamma distribution (for a distribution with no offset i.e. for loc=0)
  gamma_mean = a * theta
  mean = loc + gamma_mean

  return mean

def get_nz_from_skews_etc(nzs, zmin, zmax, nz, mix_means, outlier_fractions, outlier_means, outlier_skews, outlier_locs,  bulk_skews, bulk_locs):

  nzbin = len(mix_means)

  for izbin in range(nzbin):
    # Convert these into the mixture_nz inputs..
    mix_mean = mix_means[izbin] # This is the mean of the whole multimodal n(z)
    b_w = 1. # Fix the arbitrary scaling of the height of the n(z)s arbitrarily
    o_w = outlier_fractions[izbin] # check this is what the ws actually do!
    o_m = outlier_means[izbin]
    b_a = Gamma_skew_to_a(bulk_skews[izbin])
    o_a = Gamma_skew_to_a(outlier_skews[izbin])

    o_mean = outlier_means[izbin]
    o_loc = outlier_locs[izbin]
    o_theta = Gamma_mean_a_loc_to_theta(o_mean, o_a, o_loc)

    b_loc = bulk_locs[izbin]
    b_mean = (mix_mean - o_w * o_mean) / b_w
    b_theta = Gamma_mean_a_loc_to_theta(b_mean, b_a, b_loc)

    #print('b_w=',b_w,' o_w=',o_w)
    print('b_theta=',b_theta,' o_theta=',o_theta)
    bulk_outliers = mixture_nz([b_w, o_w], [b_a, o_a], [b_loc, o_loc], [b_theta, b_theta])
    nzs[izbin] = bulk_outliers.gridded_nz(zmin, zmax, nz, True)

  return nzs


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
  mix_means = [0.4, 0.6, 0.8, 1.0, 1.2]
  outlier_fractions = [0.1, 0.1, 0.1, 0.1, 0.1]
  outlier_means = [0.1, 0.2, 0.3, 0.4, 0.5]
  outlier_skews = [0.9, 0.9, 0.9, 0.9, 0.9]
  outlier_locs = [0.1, 0.1, 0.1, 0.1, 0.1]
  bulk_skews = [0.9, 0.9, 0.9, 0.9, 0.9]
  bulk_locs = [0.3, 0.3, 0.3, 0.3, 0.3]

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
  print('Panic if the above doesnt say NZ_SOURCE') # Do this better!
  nbin = kernel.nbin
  #if (nbin is not len(mix_means): print('Panic! - inconsistent nbin!')
  z = kernel.z
  zmin = z[0]
  nsample = len(z)
  zmax = z[nsample-1]
  nzs = kernel.nzs # This is a hack to get something the right shape to overwrite
  nzs = get_nz_from_skews_etc(nzs, zmin, zmax, nsample, mix_means, outlier_fractions, outlier_means, outlier_skews, outlier_locs,  bulk_skews, bulk_locs)
  kernel.nzs = nzs
  T.kernels[1] = kernel

  # Check and write out the new n(z)s
  output_root = 'plots/mixture_nz_des_multiprobe_20190717_2100'
  T.plots(output_root, plot_cov=False)
  out_fits_filename = 'mixture_nz_des_multiprobe_20190717_2100.fits'
  T.to_fits(filename = out_fits_filename)
