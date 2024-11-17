import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import pyvinecopulib as pv
import pickle
from tqdm import tqdm
import xitorch.interpolate as xi
import torch.nn as nn
from joblib import Parallel, delayed


def cGC_distribution(rho, u, v, shift = 0.0, scale = 1.0):
  upper = inverse_std_normal(u).reshape(len(u), 1) - rho * inverse_std_normal(v)
  # NOTE: clone & detatch allows grad computation without memory realloc 
  lower = torch.sqrt((1 - rho ** 2).clone().detach())
  input = upper / lower
  return cdf_std_normal(input)


def minmax_unif(obs):
  '''
  An informative uniform prior whose support is same as data's
  '''
  min = torch.min(obs) - 0.001
  max = torch.max(obs) + 0.001
  log_pdfs = torch.distributions.uniform.Uniform(min, max).log_prob(obs)
  cdfs = torch.distributions.uniform.Uniform(min, max).cdf(obs)
  return cdfs, log_pdfs.exp()


def grids_cdfs(size, context, rhovec, data, extrap_tail = .1, init_dist = 'Normal', a = 1., flt = 1e-6):
      num_perm = context.shape[0]
      num_data = context.shape[1]
      num_dim = context.shape[2]

      gridmat = torch.zeros([size, num_dim])
      cdfs = torch.zeros([num_perm, size, num_dim])

      for j in range(num_dim):
        min = torch.min(data[:,j]) - extrap_tail
        max = torch.max(data[:,j]) + extrap_tail
        xgrids = torch.linspace(min, max, size)
        gridmat[:,j] = xgrids
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
                Cop = cGC_distribution(rho = rhovec[j], u = cdf, v = context[perm, k, j]).reshape(size)
                cdf = (1 - alpha(k+1)) * cdf + alpha(k+1) * Cop
                cdf = torch.clip(cdf, min=flt, max=1.+flt)
            cdfs[perm, :, j] = cdf
      return gridmat, torch.mean(cdfs, dim=0)


def Energy_Score_pytorch(beta, observations_y, simulations_Y):
        n = len(observations_y)
        m = len(simulations_Y)
        # First part |Y-y|. Gives the L2 dist scaled by power beta. Is a vector of length n/one value per location.
        diff_Y_y = torch.pow(
                torch.norm(
                    (observations_y.unsqueeze(1) - simulations_Y.unsqueeze(0)).float(), dim=2, keepdim=True).reshape(-1,1),
                beta)
        # Second part |Y-Y'|. 2* because pdist counts only once.
        diff_Y_Y = 2 * torch.pow(
            nn.functional.pdist(simulations_Y),
            beta)
        Energy = 2 * torch.mean(diff_Y_y) - torch.sum(diff_Y_Y) / (m * (m - 1))
        return Energy


def cdf_lomax(x, a):
    return 1 - (1 + x) ** (-a)


def alpha(step):
    i = step
    return torch.tensor((2 - 1 / i) * (1 / (i + 1)), dtype=torch.float32)


def torch_ecdf(torch_data):
    data = torch_data.detach().numpy()
    data = pd.DataFrame(data)
    pobs = {}
    for i in range(data.shape[1]):
        series = data.iloc[:, i].values
        pobs[i] = rankdata(series) / (len(series) + 1)
    pobs = pd.DataFrame(pobs)
    return torch.tensor(np.array(pobs), dtype=torch.float32)


def inverse_std_normal(cumulative_prob):
    cumulative_prob_doube = torch.clip(cumulative_prob.double(), 1e-6, 1 - 1e-6)
    return torch.erfinv(2 * cumulative_prob_doube - 1) * torch.sqrt(torch.tensor(2.0))


def cdf_std_normal(input):
    return torch.clamp(torch.distributions.Normal(0, 1).cdf(input), 1e-6, 1 - 1e-6)


def linear_energy_grid_search(observations, rhovec, beta=0.5, size=1000, init_dist='Normal', a=1.):
    ctxtmat = get_context(observations, rhovec, init_dist, a)
    scores = torch.zeros([observations.shape[2]])
    sams = torch.rand([100, observations.shape[2]])

    def compute_dimension(dim):
        gridmatrix, gridcdf = grids_cdfs(size, ctxtmat, rhovec, observations, init_dist=init_dist, a=a)
        inv = xi.Interp1D(gridcdf[:, dim].contiguous(), gridmatrix[:, dim].contiguous(), method="linear")
        return Energy_Score_pytorch(beta, observations[0, :, dim].reshape([-1, 1]), inv(sams[:, dim].contiguous()).reshape([-1, 1]))

    scores = torch.tensor(Parallel(n_jobs=-1)(delayed(compute_dimension)(dim) for dim in range(observations.shape[2])))


    return scores


def get_context(observations, rhovec, init_dist='Normal', a=1.):
    num_perm, num_data, num_dim = observations.shape
    context = torch.zeros([num_perm, num_data, num_dim])
    for j in range(num_dim):
        for perm in range(num_perm):
            if init_dist == 'Normal':
                cdf = torch.distributions.Normal(0, 1).cdf(observations[perm, :, j])
            elif init_dist == 'Cauchy':
                cdf = torch.distributions.Cauchy(0, 1).cdf(observations[perm, :, j])
            else:
                raise ValueError("Unsupported init_dist")
            cdf = torch.clip(cdf, 1e-6, 1 - 1e-6)
            context[perm, :, j] = cdf
    return context


def evaluate_prcopula(test_points, context, rhovec, init_dist='Normal', a=1.):
    num_evals, num_dim = test_points.shape
    num_perm, num_data, _ = context.shape
    dens = torch.zeros([num_perm, num_evals, num_dim])
    cdfs = torch.zeros([num_perm, num_evals, num_dim])

    for j in range(num_dim):
        for perm in range(num_perm):
            if init_dist == 'Normal':
                cdf = torch.distributions.Normal(0, 1).cdf(test_points[:, j])
            elif init_dist == 'Cauchy':
                cdf = torch.distributions.Cauchy(0, 1).cdf(test_points[:, j])
            else:
                raise ValueError(f"Unsupported init_dist: {init_dist}")
            cdf = torch.clip(cdf, 1e-6, 1 - 1e-6)
            cdfs[perm, :, j] = cdf

            # Compute the copula density using cGC_distribution
            copula_density = cGC_distribution(rho=rhovec[j], u=cdf, v=context[perm, :, j])
            
            # Correct density computation
            marginal_density = torch.distributions.Normal(0, 1).log_prob(test_points[:, j]).exp()
            dens[perm, :, j] = marginal_density * copula_density

    return dens, cdfs