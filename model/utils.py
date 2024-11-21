import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Uniform, Normal, StudentT, MultivariateNormal
import xitorch.interpolate as xi
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from tqdm.notebook import tqdm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import scipy.interpolate as interpolator
from scipy.stats import rankdata

import xlrd
import numpy as np
import pyvinecopulib as pv
#import pyro
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import pickle
from sklearn.datasets import load_diabetes



def cdf_lomax(x, a):
  return 1 - (1+x) ** (-a)


def pdf_lomax(x, a):
  return a * (1 + x) **(-a-1)


def alpha(step):
  #alpha value derived by (Fong et al. 2021)
  i = step
  alpha = (2 - 1/i) * (1/(i+1))
  return torch.tensor(alpha, dtype = torch.float32)


def torch_ecdf(torch_data):
    data = torch_data.detach().numpy()
    data = pd.DataFrame(data)
    pobs = {}
    for i in range(data.shape[1]):
      ticker = data.columns[i]
      series = data.iloc[:,i].values
      pobs[ticker] = rankdata(data.iloc[:,i].values)/(len(data.iloc[:,i].values)+1)
    pobs = pd.DataFrame(pobs)
    pobs = np.array(pobs)
    if torch.isnan(torch.tensor(pobs).reshape(len(torch_data))).any():
      print('Error: NaN in empirical cdf')

    return torch.tensor(pobs).reshape(len(torch_data))


class TweakedUniform(torch.distributions.Uniform):
    def log_prob(self, value, context):
        return sum_except_batch(super().log_prob(value))
        # result = super().log_prob(value)
        # if len(result.shape) == 2 and result.shape[1] == 1:
        #     return result.reshape(-1)
        # else:
        #     return result

    def sample(self, num_samples, context):
        return super().sample((num_samples, ))


def is_int(x):
    return isinstance(x, int)


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def inverse_std_normal(cumulative_prob):
	'''
	Inverse of the standard normal CDF.
	'''
	cumulative_prob_doube = torch.clip(cumulative_prob.double(),1e-6,1- (1e-6))
	return torch.erfinv(2 * cumulative_prob_doube - 1) * torch.sqrt(torch.tensor(2.0))


def cdf_std_normal(input):
    return torch.clamp(torch.distributions.normal.Normal(loc = 0, scale = 1).cdf(input),1e-6,1- (1e-6))


def pdf_std_normal(input):
    return torch.distributions.normal.Normal(loc = 0, scale = 1).log_prob(input).exp()


def bvn_density(rho, u, v, shift = 0.0, scale = 1.0):

    if len(u) != len(v):
        print('Error: length of u and v should be equal')
    else:
        mean = torch.tensor([shift, shift])
        covariance_matrix = torch.tensor([[scale, rho], [rho, scale]])
        multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)

        l = len(u)
        input = torch.cat([u.reshape(l, 1), v.reshape(l, 1)], dim=1)

    return multivariate_normal.log_prob(inverse_std_normal(input)).exp()


def GC_density(rho, u, v, shift = 0.0, scale = 1.0):
    v_d = pdf_std_normal(inverse_std_normal(v)).reshape(len(v), 1)
    u_d = pdf_std_normal(inverse_std_normal(u)).reshape(len(u), 1)
    low = u_d * v_d
    up = bvn_density(rho = rho, u = u, v = v).reshape(len(u), 1)
    return up / low


def cbvn_density(rho, u, v, shift = 0.0, scale = 1.0):
   mean = torch.tensor([shift, shift])
   covariance_matrix = torch.tensor([[scale, rho], [rho, scale]])
   multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
   l = len(u)
   input = torch.cat([u.reshape(l, 1), v * torch.ones(l, 1)], dim=1)
   return multivariate_normal.log_prob(inverse_std_normal(input)).exp()


def cGC_density(rho, u, v, shift = 0.0, scale = 1.0):
    l = len(u)
    v_d = pdf_std_normal(inverse_std_normal(v))
    u_d = pdf_std_normal(inverse_std_normal(u)).reshape(l, 1)
    low = u_d * v_d
    up = cbvn_density(rho = rho, u = u, v = v).reshape(l, 1)
    return up / low


def cGC_distribution(rho, u, v, shift = 0.0, scale = 1.0):
    upper = inverse_std_normal(u).reshape(len(u), 1) - rho * inverse_std_normal(v)
    lower = torch.sqrt((1 - rho ** 2).clone().detach())
    input = upper / lower
    return cdf_std_normal(input)


#Drop highly correlated variables
def drop_corr(y, threshold = 0.98):
    data = pd.DataFrame(y)
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    y = data.drop(columns = to_drop).values
    return(y)


def create_permutatons(obs, perms):
    permutations = []
    L = obs.shape[0]
    for _ in range(perms):
        permutation = torch.randperm(L)
        sequence = obs[permutation, :]
        permutations.append(sequence)
    return torch.stack(permutations)


def batched_permutatons(obs, perms):
    permutations = []
    L = obs.shape[0]
    dims = obs.shape[1]
    for _ in range(perms):
        permutation = torch.randperm(L)
        for d in range(dims):
            sequence = obs[permutation, d]
            permutations.append(sequence)
    return torch.stack(permutations)


def bayestcop(input, latents, priordeg, priorcov):
    num_data = latents.shape[0]
    num_dim = latents.shape[1]

    postdegs = num_data + priordeg
    postcov = priorcov
    for n in range(num_data):
        postcov =+ torch.matmul(latents[n,:].reshape([1,num_dim]), latents[n,:].reshape([num_dim,1]))
    postcov = postcov /  (postdegs-num_dim+1)

    return


def Energy_Score_pytorch(beta, observations_y, simulations_Y):
    n = len(observations_y)
    m = len(simulations_Y)
    # First part |Y-y|. Gives the L2 dist scaled by power beta. Is a vector of length n/one value per location.
    diff_Y_y = torch.pow(
                        torch.norm(
                            (observations_y.unsqueeze(1) - simulations_Y.unsqueeze(0)).float(),
                            dim=2,keepdim=True).reshape(-1,1),
                        beta)
    # Second part |Y-Y'|. 2* because pdist counts only once.
    diff_Y_Y = 2 * torch.pow(
                            nn.functional.pdist(simulations_Y),
                            beta)
    Energy = 2 * torch.mean(diff_Y_y) - torch.sum(diff_Y_Y) / (m * (m - 1))
    return Energy


def simulate_GMM(d, K = 2, n = 100, n_test = 1000):
    #Simulate from d-dimensional diagonal GMM
    np.random.seed(100)
    mu = np.array([[2] * d,[-1] * d])
    sigma2 = np.ones((K, d))
    z = stats.bernoulli.rvs(p = 0.5, size = n)
    y = np.random.randn(n, d) * np.sqrt(sigma2[z, :]) + mu[z, :]
    mean_norm = np.mean(y, axis = 0)
    std_norm = np.std(y, axis = 0)
    y = (y- mean_norm) / std_norm
    z_test = stats.bernoulli.rvs(p = 0.5, size = n_test)
    y_test = np.random.randn(n_test, d) * np.sqrt(sigma2[z_test, :]) + mu[z_test, :]
    y_test = (y_test - mean_norm) / std_norm #normalize test data to have 0 mean, 1 std
    y = torch.tensor(y, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32)
    return y,y_test


def minmax_unif(obs):
    """
    An informative uniform prior whose support is same as data's
    """
    min = torch.min(obs) - 0.001
    max = torch.max(obs) + 0.001
    log_pdfs = torch.distributions.uniform.Uniform(min, max).log_prob(obs)
    cdfs = torch.distributions.uniform.Uniform(min, max).cdf(obs)
    return cdfs, log_pdfs.exp()

def empirical_dist(obs):
    """
    An informative empirical distribution which put equal probability on each data point
    """
    N = obs.shape[0]
    return torch_ecdf(obs), torch.ones_like(obs) * (1/N)

def energy_cv(data, K, up = 4., low = 2., size = 10, beta = .5):
    kfold = KFold(n_splits=K, random_state=100, shuffle=True)
    bgrids = np.linspace(low, up, size)
    in_sample = torch.zeros([size, K])
    for train, test in kfold.split(data):
        i = 0
        for epoch in tqdm(range(size)):
            controls = pv.FitControlsVinecop(
                                        family_set=[pv.BicopFamily.tll], 
                                        selection_criterion='mbic', 
                                        nonparametric_method='constant', 
                                        nonparametric_mult=bgrids[epoch], 
                                        num_threads = 2048
                                        )
            cop = pv.Vinecop(data[train], controls=controls)
            news = cop.simulate(100)
            in_sample[epoch, i] = Energy_Score_pytorch(beta, data[test], torch.tensor(news, dtype=torch.float32))
            i = i + 1
    in_sample_err = torch.mean(in_sample, dim=1)
    return bgrids[torch.argmin(in_sample_err)]


def get_context(observations, rhovec, init_dist = 'Normal', a = 1.):
    flt = 1e-6
    num_perm = observations.shape[0]
    num_data = observations.shape[1]
    num_dim = observations.shape[2]
    context = torch.zeros([num_perm, num_data, num_dim])
    for j in range(num_dim):
      for perm in range(num_perm):
        if init_dist == 'Normal':
          cdf = torch.distributions.normal.Normal(loc=0, scale=1.0).cdf(observations[perm,:,j]).reshape(num_data)
        if init_dist == 'Cauchy':
          cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(observations[perm,:,j]).reshape(num_data)
        if init_dist == 'Lomax':
          cdf = cdf_lomax(observations[perm,:,j], a)
        if init_dist == 'Unif':
          cdf, _ = minmax_unif(observations[perm,:,j].reshape(num_data))
        cdf = torch.clip(cdf, min=flt, max=1.+flt)
        context[perm, 0, j] = cdf[0]
        for k in range(1, num_data):
          Cop = cGC_distribution(rho = rhovec[j], u = cdf[1:], v = cdf[0]).reshape(num_data-k)
          cdf = (1 - alpha(k)) * cdf[1:] + alpha(k) * Cop
          cdf = torch.clip(cdf, min=flt, max=1.+flt)
          context[perm, k, j] = cdf[0]
    return context


def evaluate_prcopula(test_points, context, rhovec, init_dist = 'Normal', a = 1.):
      flt = 1e-6
      num_evals = test_points.shape[0]
      num_perm = context.shape[0]
      num_data = context.shape[1]
      num_dim = test_points.shape[1]

      dens = torch.zeros([num_perm, num_evals, num_dim])
      cdfs = torch.zeros([num_perm, num_evals, num_dim])

      for dim in tqdm(range(num_dim)):
        for perm in range(num_perm):
            if init_dist == 'Normal':
                cdf = torch.distributions.normal.Normal(loc=0, scale=1).cdf(test_points[:, dim]).reshape(num_evals)
                pdf = torch.distributions.normal.Normal(loc=0, scale=1).log_prob(test_points[:, dim]).exp().reshape(num_evals)
            if init_dist == 'Cauchy':
                cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(test_points[:, dim]).reshape(num_evals)
                pdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).log_prob(test_points[:, dim]).exp().reshape(num_evals)
            if init_dist == 'Lomax':
                cdf = cdf_lomax(test_points[:, dim], a)
                pdf = pdf_lomax(test_points[:, dim], a)
            if init_dist == 'Unif':
                cdf, pdf = minmax_unif(test_points[:, dim].reshape(num_evals))
            cdf = torch.clip(cdf, min=flt, max=1.+flt)
            for k in range(0, num_data):
                cop = cGC_density(rho = rhovec[dim], u = cdf, v = context[perm, k, dim]).reshape(num_evals)
                Cop = cGC_distribution(rho = rhovec[dim], u = cdf, v = context[perm, k, dim]).reshape(num_evals)
                cdf = (1 - alpha(k+1)) * cdf + alpha(k+1) * Cop
                cdf = torch.clip(cdf, min=flt, max=1.+flt)
                pdf = (1 - alpha(k+1)) * pdf + alpha(k+1) * cop * pdf
            dens[perm, :, dim] = pdf
            cdfs[perm, :, dim] = cdf
      return torch.mean(dens, dim=0), torch.mean(cdfs, dim=0)


def grids_cdfs(size, context, rhovec, data, extrap_tail = .1, init_dist = 'Normal', a = 1.):
    flt = 1e-6
    num_perm = context.shape[0]
    num_data = context.shape[1]
    num_dim = context.shape[2]
    gridmat = torch.zeros([size, num_dim])
    cdfs = torch.zeros([num_perm, size, num_dim])

    for dim in range(num_dim):
        min = torch.min(data[:, dim]) - extrap_tail
        max = torch.max(data[:, dim]) + extrap_tail
        xgrids = torch.linspace(min, max, size)
        gridmat[:, dim] = xgrids
        for perm in range(num_perm):
            if init_dist == 'Normal':
                cdf = torch.distributions.normal.Normal(loc=0, scale=1).cdf(xgrids).reshape(size)
            if init_dist == 'Cauchy':
                cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(xgrids).reshape(size)
            if init_dist == 'Lomax':
                cdf = cdf_lomax(xgrids, a)
            if init_dist == 'Unif':
                cdf, _ = minmax_unif(xgrids.reshape(size))
            cdf = torch.clip(cdf, min=flt, max=1.+flt)
            for k in range(0, num_data):
                Cop = cGC_distribution(rho = rhovec[dim], u = cdf, v = context[perm, k, dim]).reshape(size)
                cdf = (1 - alpha(k+1)) * cdf + alpha(k+1) * Cop
                cdf = torch.clip(cdf, min=flt, max=1.+flt)
            cdfs[perm, :, dim] = cdf
    return gridmat, torch.mean(cdfs, dim=0)


def linear_energy_grid_search(observations, rhovec, beta = 0.5, size = 1000, evalsz = 100, extrap_tail = .1, extrap_bound = .5, init_dist = 'Normal', a = 1.):
    """
    Grid search optimization by Energy Score
    """
    ctxtmat = get_context(observations, rhovec, init_dist, a)
    gridmatrix, gridcdf = grids_cdfs(size, ctxtmat, rhovec, observations, extrap_tail, init_dist, a)
    sams = torch.rand([evalsz, observations.shape[2]])
    scores = torch.zeros([observations.shape[2]])
    for dim in range(observations.shape[2]):
        lcb = torch.min(gridmatrix[:,dim].reshape([gridmatrix.shape[0]])) - extrap_bound
        ucb = torch.max(gridmatrix[:,dim].reshape([gridmatrix.shape[0]])) + extrap_bound
        sorted_grids = torch.cat([lcb.unsqueeze(0), gridmatrix[:,dim].reshape([gridmatrix.shape[0]]), ucb.unsqueeze(0)])
        cdf_values = torch.cat([torch.tensor(0.0).unsqueeze(0), gridcdf[:,dim].reshape([gridcdf.shape[0]]), torch.tensor(1.0).unsqueeze(0)])
        inv = xi.Interp1D(cdf_values, sorted_grids, method="linear")
        scores[dim] = Energy_Score_pytorch(
                                        beta, 
                                        observations[0, :, dim].reshape([observations.shape[1], 1]), 
                                        inv(sams[:,dim]).reshape([evalsz, 1])
                                        )
    return scores


def extract_grids_search(scores, lower = 0.1, upper = 0.97):
    """
    Get the optimal theta for each marginal
    """
    size = scores.shape[0]
    num_dim = scores.shape[1]
    theta_dic = torch.linspace(lower, upper, size)
    optimums = torch.zeros([num_dim])
    for dim in range(num_dim):
        interim = scores[:,dim].reshape([size])
        optimums[dim] = theta_dic[torch.argmin(interim)]
    return optimums


def linvsampling(observations, context, sams, rhovec, beta = 0.5, approx = 1000, extrap_tail = .1, extrap_bound = .5, init_dist = 'Normal', a = 1.):
    gridmatrix, gridcdf = grids_cdfs(approx, context, rhovec, observations, extrap_tail, init_dist, a)
    for dim in range(observations.shape[2]):
        lcb = torch.min(gridmatrix[:,dim].reshape([gridmatrix.shape[0]])) - extrap_bound
        ucb = torch.max(gridmatrix[:,dim].reshape([gridmatrix.shape[0]])) + extrap_bound
        sorted_grids = torch.cat([lcb.unsqueeze(0), gridmatrix[:,dim].reshape([gridmatrix.shape[0]]), ucb.unsqueeze(0)])
        cdf_values = torch.cat([torch.tensor(0.0).unsqueeze(0), gridcdf[:,dim].reshape([gridcdf.shape[0]]), torch.tensor(1.0).unsqueeze(0)])
        inv = xi.Interp1D(cdf_values, sorted_grids, method="linear")
        sams[:,dim] = inv(sams[:,dim])
    return sams