
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
  upper = inverse_std_normal(u) - rho * inverse_std_normal(v)
  #upper = inverse_std_normal(u).reshape(len(u), 1) - rho * inverse_std_normal(v)
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


def grids_cdfs(size, cdfs, rhovec, data, extrap_tail = .1, init_dist = 'Normal', a = 1., flt = 1e-6):
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
    ctxtmat = get_CDFs(observations=observations, init_dist=init_dist, a=a)
    scores = torch.zeros([observations.shape[2]])
    sams = torch.rand([100, observations.shape[2]])

    def compute_dimension(dim):
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


def get_CDFs(observations, init_dist='Normal', a=1.):
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


def evaluate_prcopula(obs, cdfs, vec_of_rho, init_dist='Normal', a=1.):
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("qbvine_training.log"),
        logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)


class QBV:
    def __init__(self, init_dist='Cauchy', perm_count=10, train_frac=0.5, seed=None):
        self.init_dist = init_dist
        self.perm_count = perm_count
        self.train_frac = train_frac
        self.seed = seed
        self.model_params = {}
        self.cdfs = None
        self.minmax = None

    def _minmax(self, data):
        """
        Apply MinMax scaling separately to features (X) and target (y).
        """
        logger.info("Applying MinMax scaling on features (X) and target (y).")

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        
        self.minmax_X = MinMaxScaler(feature_range=(0.00001, 0.99999))
        self.minmax_y = MinMaxScaler(feature_range=(0.00001, 0.99999))
        
        X_scaled = self.minmax_X.fit_transform(X)
        y_scaled = self.minmax_y.fit_transform(y.values.reshape(-1, 1))

        scaled_data = np.hstack((X_scaled, y_scaled))

        return scaled_data


    def _initialise_training_data(self, data):
        logger.info("Starting data preprocessing.")
        if data.isnull().values.any():
            raise ValueError("Data contains missing values.")
        if data.empty:
            raise ValueError("Data is empty.")
        if data.shape[0] < 2:
            raise ValueError("Data has less than 2 rows.")
        if data.shape[1] < 2:
            raise ValueError("Data has less than 2 columns.")
        

        # TODO: this has lookahead bias!
        data = self._minmax(data)
        train_data, test_data = train_test_split(data, train_size=self.train_frac, random_state=self.seed)

        logger.info("Data preprocessing completed.")
        return torch.tensor(train_data, dtype=torch.float32), torch.tensor(test_data, dtype=torch.float32)


    def fit(self, X, y, theta_iterations:int=50):
        logger.info("Starting model fitting.")
        _data = pd.concat([X, y], axis=1)
        _train, _test = self._initialise_training_data(_data)
        
        logger.info(f"Generating {self.perm_count} permutations.")
        _permutations = torch.stack(Parallel(n_jobs=-1)(
            delayed(lambda: _train[torch.randperm(_train.shape[0])])() for _ in range(self.perm_count)
            ))
        logger.info("Permutations generated.")
        
        scores_dic = self._optimise_theta(_permutations, size=theta_iterations)
        logger.info("Theta optimization completed.")
        
        optimum_thetas = self._extract_optimal_thetas(scores_dic)
        logger.info("Optimal thetas extracted.")
        
        self.cdfs = self._build_cdf_permutations(_permutations, optimum_thetas)
        logger.info("CDFs building completed.")
        
        self.model_params = self._fit_copulas(_train, optimum_thetas)
        logger.info("Model fitting completed.")


    def _optimise_theta(self, y_permutations, size:int):
        logger.info("Starting grid search for optimal theta.")
        # TODO: Shouldn't this be 0.01?
        theta_grids = torch.linspace(0.1, 0.99, size).contiguous()

        def optimize_for_grid(grid):
            logger.info(f"Optimizing for grid value: {grid}")
            return linear_energy_grid_search(y_permutations, torch.full((y_permutations.shape[2],), grid), beta=0.5, init_dist=self.init_dist)

        scores_dic = torch.stack(Parallel(n_jobs=-1)(delayed(optimize_for_grid)(grid) for grid in theta_grids))
        logger.info("Grid search completed.")
        return scores_dic


    def _extract_optimal_thetas(self, scores_dic):
        return torch.argmin(scores_dic, dim=0)


    def _build_cdf_permutations(self, _permutations, optimum_thetas):
        logger.info("Building CDFs.")
        return get_CDFs(observations=_permutations, init_dist=self.init_dist)


    def _fit_copulas(self, _data, optimum_thetas, optband_xy = 3.0):
        logger.info(f"Starting copula fitting, on data with max: {_data.max().item()} and min: {_data.min().item()}.")
        controls_xy = pv.FitControlsVinecop(
                                            family_set=[pv.BicopFamily.tll],
                                            selection_criterion='mbic',
                                            nonparametric_method='constant',
                                            nonparametric_mult=optband_xy
                                            )
        cop_xy = pv.Vinecop(_data.cpu().numpy(), controls=controls_xy)
        logger.info("Copula fitting completed.")
        return {'cop_xy': cop_xy, 'opt_rhos': optimum_thetas}


    def save_model(self, folder_name)->None:
        """
        Save the model, including the copula and necessary marginals, in the specified folder.
                folder_name/
            ├── copula.json           # Copula model in JSON format
            ├── model.pkl             # Model metadata including CDFs and parameters
            ├── min_max_scaler.pkl    # Scaler (min-max normalization)
        """
        logger.info(f"Saving model to folder: {folder_name}.")
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

        logger.info("Model saved successfully, including copula JSON and scaler.")

    @staticmethod
    def load_model(folder_name):
        """
        Load the model, including the copula and necessary marginals, from the specified folder.
        folder_name/
            ├── copula.json           # Copula model in JSON format
            ├── model.pkl             # Model metadata including CDFs and parameters
            ├── min_max_scaler.pkl    # Scaler (min-max normalization)
        """
        logger.info(f"Loading model from folder: {folder_name}.")
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
            'opt_rhos': torch.tensor(model_data['model_params']['optimum'])
            }
        model.cdfs = torch.tensor(model_data['CDFs'])
        
        logger.info("Model loaded successfully, including copula and scaler.")
        return model


    def predict(self, X):
        """
        Predict joint densities or conditional probabilities for input X.
        
        Args:
            X (DataFrame): Input test data (unseen points).

        Returns:
            Tensor: Joint densities for the test points.
        """
        
        logger.info("Starting prediction.")

        scaled_X = self.minmax_X.transform(X)

        test_points = torch.tensor(scaled_X, dtype=torch.float32)
        dens, _ = evaluate_prcopula(
                            test_points, 
                            self.cdfs, 
                            self.model_params['opt_rhos'], 
                            init_dist=self.init_dist)

        averaged_dens = dens.mean(dim=0)
        predictions = self.minmax_y.inverse_transform(averaged_dens.reshape(-1, 1))
        
        logger.info(f"Prediction completed, shape{predictions.shape} .")

        return predictions