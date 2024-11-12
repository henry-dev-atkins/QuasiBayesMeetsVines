
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
from .utils import linear_energy_grid_search, get_context, alpha, cdf_std_normal, inverse_std_normal, cdf_lomax, minmax_unif, cGC_distribution, grids_cdfs, evaluate_prcopula


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
        self.context = None
        self.mean_y = None
        self.std_y = None


    def _preprocess_data(self, data):
        logger.info("Starting data preprocessing.")
        train_data, test_data = train_test_split(data, train_size=self.train_frac, random_state=self.seed)
        logger.info("Data preprocessing completed.")
        return torch.tensor(train_data.values, dtype=torch.float32), torch.tensor(test_data.values, dtype=torch.float32)


    def fit(self, X, y):
        logger.info("Starting model fitting.")
        _data = pd.concat([X, y], axis=1)
        _train, _ = self._preprocess_data(_data)
        
        logger.info(f"Generating {self.perm_count} permutations.")
        _permutations = torch.stack(Parallel(n_jobs=-1)(
            delayed(lambda: _train[torch.randperm(_train.shape[0])])() for _ in range(self.perm_count)
            ))
        logger.info("Permutations generated.")
        
        scores_dic = self._optimize_theta(_permutations)
        logger.info("Theta optimization completed.")
        
        optimum_thetas = self._extract_optimal_thetas(scores_dic)
        logger.info("Optimal thetas extracted.")
        
        self.context = self._build_context(_permutations, optimum_thetas)
        logger.info("Context building completed.")
        
        self.model_params = self._fit_copulas(_train, _train, optimum_thetas)
        logger.info("Model fitting completed.")


    def _optimize_theta(self, y_permutations):
        logger.info("Starting grid search for optimal theta.")
        size = 50
        theta_grids = torch.linspace(0.1, 0.99, size).contiguous()

        def optimize_for_grid(grid):
            logger.debug(f"Optimizing for grid value: {grid}")
            return linear_energy_grid_search(y_permutations, torch.full((y_permutations.shape[2],), grid), beta=0.5, init_dist=self.init_dist)

        scores_dic = torch.stack(Parallel(n_jobs=-1)(delayed(optimize_for_grid)(grid) for grid in theta_grids))
        logger.info("Grid search completed.")
        return scores_dic


    def _extract_optimal_thetas(self, scores_dic):
        return torch.argmin(scores_dic, dim=0)


    def _build_context(self, _permutations, optimum_thetas):
        logger.info("Building context.")
        return get_context(_permutations, optimum_thetas, init_dist=self.init_dist)


    def _fit_copulas(self, y_train, y_test, optimum_thetas):
        logger.info(f"Starting copula fitting, on data with max: {y_train.max().item()} and min: {y_train.min().item()}.")
        optband_xy = 3.0
        controls_xy = pv.FitControlsVinecop(
                                            family_set=[pv.BicopFamily.tll],
                                            selection_criterion='mbic',
                                            nonparametric_method='constant',
                                            nonparametric_mult=optband_xy
                                            )
        cop_xy = pv.Vinecop(y_train.cpu().numpy(), controls=controls_xy)
        logger.info("Copula fitting completed.")
        return {'cop_xy': cop_xy, 'optimum': optimum_thetas}


    def save_model(self, filename):
        """
        Save the model, including the copula and necessary marginals.
        """
        logger.info(f"Saving model to {filename}.")

        copula_file = filename + '_copula.json'
        self.model_params['cop_xy'].to_json(copula_file)

        model_data = {
            'init_dist': self.init_dist,
            'perm_count': self.perm_count,
            'train_frac': self.train_frac,
            'seed': self.seed,
            'model_params': {
                'optimum': self.model_params['optimum'].tolist()
            },
            'context': self.context.cpu().numpy().tolist(),
            'mean_y': self.mean_y,
            'std_y': self.std_y,
            'copula_file': copula_file
            }

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully, including copula JSON.")



    @staticmethod
    def load_model(filename):
        """
        Load the model, including the copula and necessary marginals.
        """
        logger.info(f"Loading model from {filename}.")

        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        copula_model = pv.Vinecop.from_json(model_data['copula_file'])

        model = QBV(
            init_dist=model_data['init_dist'],
            perm_count=model_data['perm_count'],
            train_frac=model_data['train_frac'],
            seed=model_data['seed']
            )
        
        model.model_params = {
            'cop_xy': copula_model,
            'optimum': torch.tensor(model_data['model_params']['optimum'])
            }
        model.context = torch.tensor(model_data['context'])
        model.mean_y = torch.tensor(model_data['mean_y'])
        model.std_y = torch.tensor(model_data['std_y'])
        
        logger.info("Model loaded successfully, including copula.")
        return model


    def predict(self, X_test):
        """
        Predictive evaluation using Quasi-Bayesian Vine model with post-processing.
        X_test: predictors, assumed already normalized as in training.

        Returns:
        - A dictionary with:
        - 'marginal_densities': Marginal predictive densities for each feature.
        - 'joint_densities': Joint densities for each test sample.
        - 'final_scores': Combined scores incorporating copula log-likelihood.
        """
        logger.info("Starting prediction with post-processing.")
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

        # Ensure context & model parameters from training
        if self.context is None or not self.model_params:
            raise RuntimeError("Model not fitted. Fit the model before prediction.")

        rhovec = self.model_params['optimum']
        test_dens, test_cdfs = evaluate_prcopula(X_test, self.context, rhovec, init_dist=self.init_dist)

        cop_xy = self.model_params['cop_xy']
        loglik_xy = torch.tensor(cop_xy.loglik(test_cdfs.numpy()))

        # Post-processing: Compute joint densities and final scores
        joint_dens = test_dens.prod(dim=1)  # Product of marginal densities for each row
        final_score = torch.log(joint_dens) + loglik_xy  # Combine with copula log-likelihood

        logger.info("Prediction with post-processing completed.")
        return {
            'marginal_densities': test_dens,
            'joint_densities': joint_dens,
            'final_scores': final_score
            }

