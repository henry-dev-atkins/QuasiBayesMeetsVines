
import logging
from joblib import Parallel, delayed
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pyvinecopulib as pv
import xitorch.interpolate as xi
import pickle
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import time
from typing import Tuple, Dict, Any


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


def cGC_distribution(rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, shift: float = 0.0, scale: float = 1.0) -> torch.Tensor:
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
    upper = inverse_std_normal(u) - rho * inverse_std_normal(v)
    #upper = inverse_std_normal(u).reshape(len(u), 1) - rho * inverse_std_normal(v)
    # NOTE: clone & detatch allows grad computation without memory realloc 
    lower = torch.sqrt((1 - rho ** 2).clone().detach())
    input = upper / lower
    return cdf_std_normal(input)


def minmax_unif(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
                cdf, _ = minmax_unif(xgrids.reshape(size))
            cdf = torch.clip(cdf, min=flt, max=1.+flt)
            for k in range(0, num_data):
                Cop = cGC_distribution(rho = rhovec[j], u = cdf, v = cdfs[perm, k, j]).reshape(size)
                cdf = (1 - alpha(k+1)) * cdf + alpha(k+1) * Cop
                cdf = torch.clip(cdf, min=flt, max=1.+flt)
            #NOTE: changed from cdfs[perm, :, j] = cdf
            cdfs[perm, :, j] += cdf / num_data
    return gridmat, torch.mean(cdfs, dim=0)


def Energy_Score_pytorch(beta: float, observations_y: torch.Tensor, simulations_Y: torch.Tensor) -> torch.Tensor:
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
        beta)
    Energy = 2 * torch.mean(diff_Y_y) - torch.sum(diff_Y_Y) / (m * (m - 1))
    return Energy


def cdf_lomax(x: torch.Tensor, a: float) -> torch.Tensor:
    return 1 - (1 + x) ** (-a)


def alpha(step: int) -> float:
    i = step
    return torch.tensor((2 - 1 / i) * (1 / (i + 1)), dtype=torch.float32)


def torch_ecdf(torch_data: torch.Tensor) -> torch.Tensor:
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


def inverse_std_normal(cumulative_prob: torch.Tensor) -> torch.Tensor:
    cumulative_prob_doube = torch.clip(cumulative_prob.double(), 1e-6, 1 - 1e-6)
    return torch.erfinv(2 * cumulative_prob_doube - 1) * torch.sqrt(torch.tensor(2.0))


def cdf_std_normal(input: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.distributions.Normal(0, 1).cdf(input), 1e-6, 1 - 1e-6)


def linear_energy_grid_search(observations: torch.Tensor, rhovec: torch.Tensor, beta: float = 0.5, 
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
    ctxtmat = generate_CDFs(observations=observations, init_dist=init_dist, a=a)
    scores = torch.zeros([observations.shape[2]])
    sams = torch.rand([100, observations.shape[2]])

    def compute_dimension(dim: int) -> torch.Tensor:
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
        return Energy_Score_pytorch(
                                    beta,
                                    observations[0, :, dim].reshape([-1, 1]),
                                    inv(sams[:, dim].contiguous()).reshape([-1, 1])
                                    )
    scores = torch.tensor(Parallel(n_jobs=-1)(delayed(compute_dimension)(dim) for dim in range(observations.shape[2])))
    return scores


def generate_CDFs(observations: torch.Tensor, init_dist: str = 'Normal', a: float = 1.) -> torch.Tensor:
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


def evaluate_prcopula(obs: torch.Tensor, cdfs: torch.Tensor, vec_of_rho: torch.Tensor, 
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
                copula_density *= cGC_distribution(
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




########################################################################################################################################




class QBV:
    def __init__(self, init_dist='Cauchy', perm_count=10, train_frac=0.5, seed=None, verbose=1):
        self.init_dist = init_dist
        self.perm_count = perm_count
        self.train_frac = train_frac
        self.seed = seed
        self.model_params = {}
        self.cdfs = None
        self.minmax = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("qbvine_training.log"),
                logging.StreamHandler()
                ]
            )
        self.logger = logging.getLogger(__name__)


    def _minmax(self, data):
        """
        Apply MinMax scaling separately to features (X) and target (y).
        """
        self.logger.info("Applying MinMax scaling on features (X) and target (y).")

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        
        self.minmax_X = MinMaxScaler(feature_range=(0.00001, 0.99999))
        self.minmax_y = MinMaxScaler(feature_range=(0.00001, 0.99999))
        
        X_scaled = self.minmax_X.fit_transform(X)
        self.logger.debug(f"Scaled X variance: {np.var(X_scaled, axis=0)}")
        y_scaled = self.minmax_y.fit_transform(y.values.reshape(-1, 1))

        scaled_data = np.hstack((X_scaled, y_scaled))

        return scaled_data


    def _initialise_training_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        self.logger.info("Starting data preprocessing.")
        if data.isnull().values.any():
            raise ValueError("Data contains missing values.")
        if data.empty:
            raise ValueError("Data is empty.")
        if data.shape[0] < 2:
            raise ValueError("Data has less than 2 rows.")
        if data.shape[1] < 2:
            raise ValueError("Data has less than 2 columns.")
        
        # FIXME: this order of minmax then split has lookahead bias!
        data = self._minmax(data)
        train_data, test_data = train_test_split(
                                                data, 
                                                train_size = self.train_frac, 
                                                random_state = self.seed
                                                )
        self.logger.info("Data preprocessing completed.")
        return torch.tensor(train_data, dtype=torch.float32), torch.tensor(test_data, dtype=torch.float32)


    def fit(self, X: pd.DataFrame, y: pd.Series, theta_iterations: int = 50) -> None:
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
        self.logger.info("Starting model fitting.")
        start_time = time.time()
        
        _data = pd.concat([X, y], axis=1)
        _train, _test = self._initialise_training_data(_data)

        if _train.numel() == 0:
            raise ValueError("Training data is empty.")
        
        self.logger.info(f"Generating {self.perm_count} permutations.")
        perm_start = time.time()
        _permutations = torch.stack(Parallel(n_jobs=-1)(
            delayed(lambda: _train[torch.randperm(_train.shape[0])])() for _ in range(self.perm_count)
            ))
        self.logger.debug(f"Permutation variance: {torch.var(_permutations, dim=0)}")
        self.logger.info(f"Permutations generated in {time.time() - perm_start:.2f} seconds.")
        
        optimize_start = time.time()
        scores_dic = self._optimise_theta(_permutations, size=theta_iterations)
        self.logger.info(f"Theta optimization completed in {time.time() - optimize_start:.2f} seconds.")
        
        optimum_thetas = self.torch.argmin(scores_dic, dim=0)
        self.logger.debug(f"Optimal thetas: {optimum_thetas}")
        self.logger.info("Optimal thetas extracted.")
        
        cdf_start = time.time()
        self.cdfs = self._build_cdf_permutations(_permutations, optimum_thetas)
        self.logger.debug(f"CDFs: {self.cdfs}")
        self.logger.info(f"CDFs building completed in {time.time() - cdf_start:.2f} seconds.")
        
        fit_copula_start = time.time()
        self.model_params = self._fit_copulas(_train, optimum_thetas)
        self.logger.debug(f"Copula parameters: {self.model_params['cop_xy'].parameters}")
        self.logger.info(f"Model fitting completed in {time.time() - fit_copula_start:.2f} seconds.")
        return 



    def _optimise_theta(self, y_permutations: torch.Tensor, size: int) -> torch.Tensor:
        """
        Perform a grid search to optimize theta values.

        This function performs a grid search to find the optimal theta values by computing the Energy Score for each grid point. It uses parallel processing to speed up the computation.

        Parameters:
        ----------
        y_permutations : torch.Tensor
            The tensor of permuted training data with shape (num_perm, num_data, num_dim).
        size : int
            The number of grid points to generate for the grid search.

        Returns:
        -------
        torch.Tensor
            A tensor containing the Energy Scores for each grid point.

        Notes:
        -----
        - The function logs the progress and the grid values being optimized.
        - The `linear_energy_grid_search` function is used to compute the Energy Score for each grid point.
        - The `torch.linspace` function is used to generate the grid points for theta values.
        """
        self.logger.info("Starting grid search for optimal theta.")
        # TODO: Shouldn't this be 0.01?
        theta_grids = torch.linspace(0.1, 0.99, size).contiguous()

        def optimize_for_grid(grid):
            self.logger.info(f"Optimizing for grid value: {grid}")
            return linear_energy_grid_search(y_permutations, torch.full((y_permutations.shape[2],), grid), beta=0.5, init_dist=self.init_dist)

        scores_dic = torch.stack(Parallel(n_jobs=-1)(delayed(optimize_for_grid)(grid) for grid in theta_grids))
        self.logger.info("Grid search completed.")
        return scores_dic


    def _build_cdf_permutations(self, _permutations: torch.Tensor, optimum_thetas: torch.Tensor) -> torch.Tensor:
        """
        From the permutations and optimal thetas, build the CDFs.
        # TODO: Do we need the optimum_thetas to generate CDFs?
        # TODO: Should we be using the same permutations for all thetas?
        # TODO: Is this func (as it stands) necessary?
        """
        self.logger.info("Building CDFs.")
        return generate_CDFs(observations=_permutations, init_dist=self.init_dist)


    def _fit_copulas(self, _data: torch.Tensor, optimum_thetas: torch.Tensor, optband_xy: float = 3.0) -> Dict[str, Any]:
        """
        Fit copulas to the provided data using the optimal theta values.

        This function fits copulas to the provided data using the optimal theta values. It uses the pyvinecopulib library to fit a vine copula model with specified controls.

        Parameters:
        ----------
        _data : torch.Tensor
            The tensor of data to fit the copulas to.
        optimum_thetas : torch.Tensor
            The tensor of optimal theta values.
        optband_xy : float, optional
            The bandwidth parameter for the nonparametric method (default is 3.0).

        Returns:
        -------
        Dict[str, Any]
            A dictionary containing the fitted copula model and related parameters.

        Notes:
        -----
        - The function logs the progress and the range of the data being fitted.
        - The pyvinecopulib library is used to fit a vine copula model with specified controls.
        """
        self.logger.info(f"Starting copula fitting, on data with max: {_data.max().item()} and min: {_data.min().item()}.")
        controls_xy = pv.FitControlsVinecop(
                                            family_set=[pv.BicopFamily.tll],
                                            selection_criterion='mbic',
                                            nonparametric_method='constant',
                                            nonparametric_mult=optband_xy
                                            )
        cop_xy = pv.Vinecop(_data.cpu().numpy(), controls=controls_xy)
        self.logger.info("Copula fitting completed.")
        return {'cop_xy': cop_xy, 'opt_rhos': optimum_thetas}


    def save_model(self, folder_name: str) -> None:
        """
        Save the model, including the copula and necessary marginals, in the specified folder.
                folder_name/
            ├── copula.json           # Copula model in JSON format
            ├── model.pkl             # Model metadata including CDFs and parameters
            ├── min_max_scaler.pkl    # Scaler (min-max normalization)
        """
        self.logger.info(f"Saving model to folder: {folder_name}.")
        os.makedirs(folder_name, exist_ok=True)

        copula_file = os.path.join(folder_name, 'copula.json')
        model_file = os.path.join(folder_name, 'model.pkl')

        self.model_params['cop_xy'].to_json(copula_file)

        model_data = {
            'init_dist': self.init_dist,
            'perm_count': self.perm_count,
            'train_frac': self.train_frac,
            'seed': self.seed,
            'model_params': {
                            'opt_rhos': self.model_params['opt_rhos'].tolist()
                            },
            'CDFs': self.cdfs.cpu().numpy().tolist(),
            'copula_file': copula_file
            }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        scaler_X_file = os.path.join(folder_name, 'min_max_scaler_X.pkl')
        scaler_y_file = os.path.join(folder_name, 'min_max_scaler_y.pkl')

        with open(scaler_X_file, 'wb') as f:
            pickle.dump(self.minmax_X, f)
        with open(scaler_y_file, 'wb') as f:
            pickle.dump(self.minmax_y, f)

        self.logger.info("Model saved successfully, including copula JSON and scaler.")

    @staticmethod
    def load_model(folder_name: str) -> Any: # TODO: Type?
        """
        Load the model, including the copula and necessary marginals, from the specified folder.
        folder_name/
            ├── copula.json           # Copula model in JSON format
            ├── model.pkl             # Model metadata including CDFs and parameters
            ├── min_max_scaler.pkl    # Scaler (min-max normalization)
        """
        
        model_file = os.path.join(folder_name, 'model.pkl')
        scaler_X_file = os.path.join(folder_name, 'min_max_scaler_X.pkl')
        scaler_y_file = os.path.join(folder_name, 'min_max_scaler_y.pkl')

        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)

        copula_model = pv.Vinecop(model_data['copula_file'])

        model = QBV(
            init_dist=model_data['init_dist'],
            perm_count=model_data['perm_count'],
            train_frac=model_data['train_frac'],
            seed=model_data['seed']
            )
        with open(scaler_X_file, 'rb') as f:
            model.minmax_X = pickle.load(f)
        with open(scaler_y_file, 'rb') as f:
            model.minmax_y = pickle.load(f)

        model.model_params = {
            'cop_xy': copula_model,
            'opt_rhos': torch.tensor(model_data['model_params']['opt_rhos'])
            }
        model.cdfs = torch.tensor(model_data['CDFs'])
        
        model.logger.info("Model loaded successfully, including copula and scaler.")
        return model


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict joint densities or conditional probabilities for input X.
        NOTE: This is the only func that works with pandas dataframes.
        Args:
            X (pd.DataFrame): Input test data (unseen points).

        Returns:
            Tensor: Joint densities for the test points.
        """
        
        self.logger.info("Starting prediction.")

        scaled_X = self.minmax_X.transform(X)
        self.logger.debug(f"Scaled X variance: {np.var(scaled_X, axis=0)}")

        test_points = torch.tensor(scaled_X, dtype=torch.float32)
        dens, _ = evaluate_prcopula(
                            test_points, 
                            self.cdfs, 
                            self.model_params['opt_rhos'], 
                            init_dist=self.init_dist)

        averaged_dens = dens.mean(dim=0)
        predictions = self.minmax_y.inverse_transform(averaged_dens.reshape(-1, 1))
        
        self.logger.info(f"Prediction completed with shape: {predictions.shape} and values: {predictions}")

        return pd.DataFrame(predictions, columns=["Predicted Density"])