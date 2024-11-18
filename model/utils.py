
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import pyvinecopulib as pv
import pickle
from tqdm import tqdm
import xitorch.interpolate as xi
import torch.nn as nn
from joblib import Parallel, delayed


def compute_conditional_gaussian_copula  (rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, shift: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """
    Compute the conditional Gaussian copula (cGC) distribution.

    This function calculates the conditional distribution of a Gaussian copula given two sets of uniform random variables `u` and `v`, and a correlation coefficient `rho`.

    Parameters:
    ----------
    rho : torch.Tensor
        The correlation coefficient tensor between the variables.
    u : torch.Tensor
        The first set of uniform random variables.
    v : torch.Tensor
        The second set of uniform random variables.
    shift : float, optional
        A shift parameter for the distribution (default is 0.0).
    scale : float, optional
        A scale parameter for the distribution (default is 1.0).

    Returns:
    -------
    torch.Tensor
        The conditional Gaussian copula distribution values.

    Notes:
    -----
    - The function uses the inverse of the standard normal CDF to transform the uniform random variables `u` and `v` to the normal space.
    - The `clone().detach()` method is used to allow gradient computation without memory reallocation.
    """
    upper = inverse_standard_normal_cdf(u) - rho * inverse_standard_normal_cdf(v)
    #upper = inverse_std_normal(u).reshape(len(u), 1) - rho * inverse_std_normal(v)
    # NOTE: clone & detatch allows grad computation without memory realloc 
    lower = torch.sqrt((1 - rho ** 2).clone().detach())
    input = upper / lower
    return standard_normal_cdf(input)


def uniform_cdf_pdf_bounds (obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the CDF and PDF of a uniform distribution with support based on the observed data.

    This function calculates the cumulative distribution function (CDF) and the probability density function (PDF) of a uniform distribution whose support is slightly extended beyond the minimum and maximum values of the observed data.

    Parameters:
    ----------
    obs : torch.Tensor
        The observed data tensor.

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - cdfs: The cumulative distribution function values for the observed data.
        - pdfs: The probability density function values for the observed data.

    Notes:
    -----
    - The support of the uniform distribution is extended by 0.001 beyond the minimum and maximum values of the observed data to avoid boundary issues.
    """
    min = torch.min(obs) - 0.001
    max = torch.max(obs) + 0.001
    log_pdfs = torch.distributions.uniform.Uniform(min, max).log_prob(obs)
    cdfs = torch.distributions.uniform.Uniform(min, max).cdf(obs)
    return cdfs, log_pdfs.exp()


def grids_cdfs(size: int, cdfs: torch.Tensor, rhovec: torch.Tensor, data: torch.Tensor, 
               extrap_tail: float = 0.1, init_dist: str = 'Normal', a: float = 1., flt: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate grid points and corresponding CDFs for given data and parameters.

    This function generates grid points and computes the cumulative distribution functions (CDFs) for each dimension of the data, based on the specified initial distribution and correlation coefficients.

    Parameters:
    ----------
    size : int
        The number of grid points to generate.
    cdfs : torch.Tensor
        The initial CDF values for the data.
    rhovec : torch.Tensor
        The correlation coefficients for the copula.
    data : torch.Tensor
        The observed data tensor.
    extrap_tail : float, optional
        The amount to extend the support of the uniform distribution beyond the minimum and maximum values of the observed data (default is 0.1).
    init_dist : str, optional
        The initial distribution to use ('Normal', 'Cauchy', 'Lomax', 'Unif') (default is 'Normal').
    a : float, optional
        The shape parameter for the Lomax distribution (default is 1.0).
    flt : float, optional
        A small value to clip the CDF values to avoid boundary issues (default is 1e-6).

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - gridmat: The generated grid points for each dimension.
        - mean_cdfs: The mean CDF values across permutations.

    Notes:
    -----
    - The support of the uniform distribution is extended by `extrap_tail` beyond the minimum and maximum values of the observed data to avoid boundary issues.
    - The function supports different initial distributions: 'Normal', 'Cauchy', 'Lomax', and 'Unif'.
    - The CDF values are clipped to the range [flt, 1.0 + flt] to avoid boundary issues.
    """
    num_perm = cdfs.shape[0]
    num_data = cdfs.shape[1]
    num_dim = cdfs.shape[2]

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
                cdf, _ = uniform_cdf_pdf_bounds(xgrids.reshape(size))
            cdf = torch.clip(cdf, min=flt, max=1.+flt)
            for k in range(0, num_data):
                Cop = compute_conditional_gaussian_copula(rho = rhovec[j], u = cdf, v = cdfs[perm, k, j]).reshape(size)
                cdf = (1 - alpha(k+1)) * cdf + alpha(k+1) * Cop
                cdf = torch.clip(cdf, min=flt, max=1.+flt)
            #NOTE: changed from cdfs[perm, :, j] = cdf
            cdfs[perm, :, j] += cdf / num_data
    return gridmat, torch.mean(cdfs, dim=0)


def energy_score(beta: float, observations_y: torch.Tensor, simulations_Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Energy Score for a set of observations and simulations.

    This function calculates the Energy Score, which is a measure of the accuracy of probabilistic predictions. It is based on the Euclidean distance between observations and simulations, scaled by a power parameter `beta`.

    Parameters:
    ----------
    beta : float
        The power parameter for scaling the Euclidean distance.
    observations_y : torch.Tensor
        The tensor of observed values.
    simulations_Y : torch.Tensor
        The tensor of simulated values.

    Returns:
    -------
    torch.Tensor
        The computed Energy Score.

    Notes:
    -----
    - The Energy Score is computed as:
      2 * mean(|Y - y|^beta) - mean(|Y - Y'|^beta)
      where Y and Y' are independent samples from the predictive distribution.
    - The `pdist` function from `torch.nn.functional` is used to compute pairwise distances between simulations.
    """
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
                            beta
    )
    Energy = 2 * torch.mean(diff_Y_y) - torch.sum(diff_Y_Y) / (m * (m - 1))
    return Energy


def cdf_lomax(x: torch.Tensor, a: float) -> torch.Tensor:
    return 1 - (1 + x) ** (-a)


def alpha(step: int) -> float:
    i = step
    return torch.tensor((2 - 1 / i) * (1 / (i + 1)), dtype=torch.float32)


def empirical_cdf (torch_data: torch.Tensor) -> torch.Tensor:
    """
    Compute the empirical cumulative distribution function (ECDF) for each column of the input tensor.

    This function calculates the ECDF for each column of the input tensor and returns the result as a tensor.

    Parameters:
    ----------
    torch_data : torch.Tensor
        The input data tensor.

    Returns:
    -------
    torch.Tensor
        A tensor containing the ECDF values for each column of the input tensor.

    Notes:
    -----
    - The input tensor is converted to a NumPy array and then to a pandas DataFrame for easier manipulation.
    - The `rankdata` function from `scipy.stats` is used to compute the ECDF values.
    - The ECDF values are scaled by dividing by the number of observations plus one to ensure they are in the range (0, 1).
    """
    data = torch_data.detach().numpy()
    data = pd.DataFrame(data)
    pobs = {}
    for i in range(data.shape[1]):
        series = data.iloc[:, i].values
        pobs[i] = rankdata(series) / (len(series) + 1)
    pobs = pd.DataFrame(pobs)
    return torch.tensor(np.array(pobs), dtype=torch.float32)


def inverse_standard_normal_cdf (cumulative_prob: torch.Tensor) -> torch.Tensor:
    cumulative_prob_doube = torch.clip(cumulative_prob.double(), 1e-6, 1 - 1e-6)
    return torch.erfinv(2 * cumulative_prob_doube - 1) * torch.sqrt(torch.tensor(2.0))


def standard_normal_cdf (input: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.distributions.Normal(0, 1).cdf(input), 1e-6, 1 - 1e-6)


def energy_score_grid_search (observations: torch.Tensor, rhovec: torch.Tensor, beta: float = 0.5,
                              size: int = 1000, init_dist: str = 'Normal', a: float = 1.) -> torch.Tensor:
    """
    Perform a grid search to compute the Energy Score for each dimension of the observations.

    This function generates grid points and computes the cumulative distribution functions (CDFs) for each dimension of the observations. It then interpolates these CDFs and calculates the Energy Score for each dimension using the specified parameters.

    Parameters:
    ----------
    observations : torch.Tensor
        The observed data tensor with shape (num_perm, num_data, num_dim).
    rhovec : torch.Tensor
        The correlation coefficients tensor for the copula.
    beta : float, optional
        The power parameter for scaling the Euclidean distance in the Energy Score (default is 0.5).
    size : int, optional
        The number of grid points to generate (default is 1000).
    init_dist : str, optional
        The initial distribution to use ('Normal' or 'Cauchy') (default is 'Normal').
    a : float, optional
        The shape parameter for the Lomax distribution (default is 1.0).

    Returns:
    -------
    torch.Tensor
        The computed Energy Scores for each dimension.

    Notes:
    -----
    - The function uses the `generate_CDFs` function to compute the initial CDFs for the observations.
    - The `grids_cdfs` function is used to generate grid points and corresponding CDFs.
    - The `xi.Interp1D` function is used to interpolate the CDFs.
    - The `Energy_Score_pytorch` function is used to compute the Energy Score for each dimension.
    """
    ctxtmat = build_initial_cdfs(observations=observations, init_dist=init_dist, a=a)
    scores = torch.zeros([observations.shape[2]])
    sams = torch.rand([100, observations.shape[2]])

    def process_dimension(dim: int) -> torch.Tensor:
        gridmatrix, gridcdf = grids_cdfs(
                                        size,
                                        ctxtmat,
                                        rhovec,
                                        observations,
                                        init_dist=init_dist,
                                        a=a
                                        )
        inv = xi.Interp1D(
                        gridcdf[:, dim].contiguous(),
                        gridmatrix[:, dim].contiguous(),
                        method="linear"
                        )
        return energy_score(
                                    beta,
                                    observations[0, :, dim].reshape([-1, 1]),
                                    inv(sams[:, dim].contiguous()).reshape([-1, 1])
                                    )
    scores = torch.tensor(Parallel(n_jobs=-1)(delayed(process_dimension)(dim) for dim in range(observations.shape[2])))
    return scores


def build_initial_cdfs (observations: torch.Tensor, init_dist: str = 'Normal', a: float = 1.) -> torch.Tensor:
    """
    Evaluate the probabilistic copula model for given observations and CDFs.

    This function calculates the densities and CDFs for a probabilistic copula model based on the given observations, initial CDFs, and correlation coefficients.

    Parameters:
    ----------
    obs : torch.Tensor
        The observed data tensor with shape (num_evals, num_dim).
    cdfs : torch.Tensor
        The initial CDF values tensor with shape (num_perm, num_data, num_dim).
    vec_of_rho : torch.Tensor
        The correlation coefficients tensor for the copula.
    init_dist : str, optional
        The initial distribution to use ('Normal' or 'Cauchy') (default is 'Normal').
    a : float, optional
        The shape parameter for the Lomax distribution (default is 1.0).

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - densities: The computed densities for each permutation and evaluation.
        - cdfs: The computed CDFs for each permutation and evaluation.

    Notes:
    -----
    - The function supports different initial distributions: 'Normal' and 'Cauchy'.
    - The CDF values are clipped to the range [1e-6, 1 - 1e-6] to avoid boundary issues.
    - The joint copula density is computed across dimensions for each permutation.
    """
    # TODO: Should there be a theta/rho/correlation consideration here?
    num_perm, num_data, num_dim = observations.shape
    cdfs = torch.zeros([num_perm, num_data, num_dim])
    for j in range(num_dim):
        for perm in range(num_perm):
            if init_dist == 'Normal':
                cdf = torch.distributions.Normal(0, 1).cdf(observations[perm, :, j])
            elif init_dist == 'Cauchy':
                cdf = torch.distributions.Cauchy(0, 1).cdf(observations[perm, :, j])
            else:
                raise ValueError("Unsupported init_dist")
            cdf = torch.clip(cdf, 1e-6, 1 - 1e-6)
            cdfs[perm, :, j] = cdf
    return cdfs


def evaluate_predictive_copula (obs: torch.Tensor, cdfs: torch.Tensor, vec_of_rho: torch.Tensor, 
                      init_dist: str = 'Normal', a: float = 1.) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the probabilistic copula model for given observations and CDFs.

    This function calculates the densities and CDFs for a probabilistic copula model based on the given observations, initial CDFs, and correlation coefficients.

    Parameters:
    ----------
    obs : torch.Tensor
        The observed data tensor with shape (num_evals, num_dim).
    cdfs : torch.Tensor
        The initial CDF values tensor with shape (num_perm, num_data, num_dim).
    vec_of_rho : torch.Tensor
        The correlation coefficients tensor for the copula.
    init_dist : str, optional
        The initial distribution to use ('Normal' or 'Cauchy') (default is 'Normal').
    a : float, optional
        The shape parameter for the Lomax distribution (default is 1.0).

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - densities: The computed densities for each permutation and evaluation.
        - cdfs: The computed CDFs for each permutation and evaluation.

    Notes:
    -----
    - The function supports different initial distributions: 'Normal' and 'Cauchy'.
    - The CDF values are clipped to the range [1e-6, 1 - 1e-6] to avoid boundary issues.
    - The joint copula density is computed across dimensions for each permutation.
    """
    num_evals, num_dim = obs.shape
    num_perm, num_data, _ = cdfs.shape
    densities = torch.zeros([num_perm, num_evals])
    cdfs = torch.zeros([num_perm, num_evals, num_dim])

    for perm in range(num_perm):
        for j in range(num_dim):
            if init_dist == 'Normal':
                marginal_dist = torch.distributions.Normal(0, 1)
            elif init_dist == 'Cauchy':
                marginal_dist = torch.distributions.Cauchy(0, 1)
            else:
                raise ValueError(f"Unsupported init_dist: {init_dist}")

            cdf = marginal_dist.cdf(obs[:, j])
            cdf = torch.clip(cdf, 1e-6, 1 - 1e-6)
            cdfs[perm, :, j] = cdf

            marginal_density = marginal_dist.log_prob(obs[:, j]).exp()

            if j == 0:
                densities[perm, :] = marginal_density
            else:
                densities[perm, :] *= marginal_density

        # Compute the joint copula density across dimensions for this permutation
        copula_density = 1.0
        for j in range(num_dim):
            for k in range(j + 1, num_dim):
                # Pass scalars or per-sample pairs, not vectors.
                copula_density *= compute_conditional_gaussian_copula(
                                                                    rho=vec_of_rho[j],
                                                                    u=cdfs[perm, :, j],
                                                                    v=cdfs[perm, :, k]
                                                                    )

        densities[perm, :] *= copula_density
    print(f"dens shape: {densities.shape}")
    print(f"cdf shape: {cdfs.shape}")
    print(f"obs shape: {obs.shape}")
    print(f"vec_of_rho shape: {vec_of_rho.shape}")

    print(f"a column of densities: {densities[0, :].mean()}, {densities[0, :].std()}")
    print(f"a row of densities: {densities[:, 0].mean()}, {densities[:, 0].std()}")
    print(f"densities[0]: {densities[0]}")
    print(f"densities[0]: {densities[0].mean()}, {densities[0].std()}")
    return densities, cdfs