import logging
from joblib import Parallel, delayed
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pyvinecopulib as pv
import pickle
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import time
from typing import Tuple, Dict, Any

from model.utils import energy_score_grid_search, build_initial_cdfs, evaluate_predictive_copula

class QBV:
    """
    Main class for training and prediction using the Quasi-Bayes-Vine (QBV) model.
    """
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

        x, y = data.iloc[:, :-1], data.iloc[:, -1]
        
        # TODO: Review float32 vs closeness to 0 here.
        self.minmax_X = MinMaxScaler(feature_range=(0.00001, 0.99999))
        self.minmax_y = MinMaxScaler(feature_range=(0.00001, 0.99999))
        
        x_scaled = self.minmax_X.fit_transform(x)
        self.logger.debug(f"Scaled X variance: {np.var(x_scaled, axis=0)}")
        y_scaled = self.minmax_y.fit_transform(y.values.reshape(-1, 1))

        scaled_data = np.hstack((x_scaled, y_scaled))

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
        
        _data = pd.concat([X, y], axis=1)
        _train, _test = self._initialise_training_data(_data)

        if _train.numel() == 0:
            raise ValueError("Training data is empty.")
        
        self.logger.info(f"Generating {self.perm_count} permutations.")
        perm_start = time.time()
        _permutations = torch.stack(
            Parallel(n_jobs=-1)(delayed(lambda: _train[torch.randperm(_train.shape[0])])() for _ in range(self.perm_count)
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
            return energy_score_grid_search(y_permutations, torch.full((y_permutations.shape[2],), grid), init_dist=self.init_dist)

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
        return build_initial_cdfs(observations=_permutations, init_dist=self.init_dist)


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

        scaler_x_file = os.path.join(folder_name, 'min_max_scaler_X.pkl')
        scaler_y_file = os.path.join(folder_name, 'min_max_scaler_y.pkl')

        with open(scaler_x_file, 'wb') as f:
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
        dens, _ = evaluate_predictive_copula (
                            test_points, 
                            self.cdfs, 
                            self.model_params['opt_rhos'], 
                            init_dist=self.init_dist)

        # Can we assume mean will work if dens is not gaussian?
        averaged_dens = dens.mean(dim=0)
        predictions = self.minmax_y.inverse_transform(averaged_dens.reshape(-1, 1))

        #  Add the R-BP step! 
        
        self.logger.info(f"Prediction completed with shape: {predictions.shape} and values: {predictions}")

        return pd.DataFrame(predictions, columns=["Predicted Density"])