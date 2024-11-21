
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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from tqdm import tqdm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import scipy.interpolate as interpolator
from scipy.stats import rankdata

import xlrd
import numpy as np
import pyvinecopulib as pv
from sklearn.base import BaseEstimator, RegressorMixin
import pickle
from sklearn.datasets import load_diabetes
import time

from .utils import (linear_energy_grid_search, extract_grids_search, 
                    energy_cv, evaluate_prcopula, drop_corr, create_permutatons, get_context)

class QBV(BaseEstimator, RegressorMixin):
    def __init__(self, p0_class='Cauchy'):
        self.p0_class = p0_class

    def optimise_correlations(self, corr_iter: int, y_permutations: torch.tensor, corr_grids: torch.tensor, y_dim: int):
        scores_dic = torch.zeros([corr_iter, y_dim])

        for grids in tqdm(range(corr_iter)):
            scores_dic[grids, :] = linear_energy_grid_search(
                observations=y_permutations,
                rhovec=torch.linspace(corr_grids[grids], corr_grids[grids], y_dim),
                init_dist=self.p0_class
            )

        opt = extract_grids_search(scores_dic, lower=0.5, upper=0.99)
        return opt

    def train_vine_copula_general(self, pseudos, K=10, up=4.0, low=2.0, size=2):
        opt_band = energy_cv(pseudos, K=K, up=up, low=low, size=size)
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.tll],
            selection_criterion='mbic',
            nonparametric_method='constant',
            nonparametric_mult=opt_band,
            num_threads=2048
        )
        copula = pv.Vinecop(pseudos, controls=controls)
        return copula, opt_band

    def evaluate(self, copula_xy, copula_x, dim_theta, ctxt, obs):
        _dens, _cdfs = evaluate_prcopula(obs, ctxt, dim_theta, init_dist=self.p0_class)
        cop_dens_xy = torch.tensor(copula_xy.loglik(_cdfs), dtype=torch.float32) / len(obs)
        cop_dens_x = torch.tensor(copula_x.loglik(_cdfs[:, :-1]), dtype=torch.float32) / len(obs)
        nll = -cop_dens_xy - torch.log((_dens[:, -1])).mean() + cop_dens_x
        return nll, _dens, _cdfs

    def _preprocess(self, in_data:pd.DataFrame):
        _data = drop_corr(in_data)
        N = np.shape(_data)[0]
        y = torch.tensor(_data, dtype=torch.float32)
        y_test = y
        return y, y_test

    def fit(self, data: pd.DataFrame, iterations=5, perms=10):
        """
        Fit the QBV model to the given data.

        Args:
        - data: Training data as a DataFrame. Last column is assumed to be the target (y).
        - iterations: Number of iterations for optimizing correlations.
        - perms: Number of permutations for pseudo-observations.
        """
        # Preprocess the data and create permutations
        y, y_test = self._preprocess(data)
        y_permutations = create_permutatons(y, perms)

        # Optimize correlations
        self.correlations = self.optimise_correlations(
            corr_iter=iterations,
            y_permutations=y_permutations,
            y_dim=y.shape[1],
            corr_grids=torch.linspace(0.1, 0.99, iterations),
        )

        # Compute context and pseudo-observations
        ctxt = get_context(
            y_permutations,
            self.correlations,
            init_dist=self.p0_class
        )
        _, pseudos = evaluate_prcopula(y, ctxt, self.correlations, init_dist=self.p0_class)

        # Train the copulas
        copula_xy, optband_xy = self.train_vine_copula_general(pseudos)
        copula_x, optband_x = self.train_vine_copula_general(pseudos[:, :-1])

        # Store trained copulas
        self.copula_xy = {'copula': copula_xy, 'bandwidth': optband_xy}
        self.copula_x = {'copula': copula_x, 'bandwidth': optband_x}

        # Compute p_y(y) via R-BP
        print("[INFO] Computing marginal density p_y using Recursive Bayesian Prediction...")
        nll, _, cdfs = self.evaluate(
            copula_xy,
            copula_x,
            self.correlations,
            ctxt,
            y_test
        )
        self.p_y = cdfs[:, -1]  # Use the final marginal from the recursive prediction
        print(f"[INFO] Marginal density p_y computed and stored. Shape: {self.p_y.shape}")

        return self

    def _trained(self):
        if not hasattr(self, 'copula_xy'):
            return False, 'copula_xy'
        if not hasattr(self, 'copula_x'):
            return False, 'copula_x'
        if not hasattr(self, 'correlations'):
            return False, 'correlations'
        return True, '_'
    

    def _likelihood_joint(self, X, y):
        """
        Compute the log-likelihood of the joint copula c(y, X).

        Args:
        - X: Input data matrix of shape [num_points, d] (a single sample replicated num_points times).
        - y: Grid of y values of shape [num_points, 1].

        Returns:
        - joint_loglik: Log-likelihood of the joint copula, shape [num_points].
        """
        num_points, d = X.shape
        assert y.shape[0] == num_points, "The number of rows in X must match the number of y grid points."

        # Combine X and y to form joint data
        joint_data = torch.cat((X, y), dim=-1)  # Shape: [num_points, d+1]

        # Compute pseudo-observations and log-likelihood
        ctxt_joint = get_context(joint_data.unsqueeze(0), self.correlations, self.p0_class)

        _, pseudo_joint = evaluate_prcopula(joint_data, ctxt_joint, self.correlations, init_dist=self.p0_class)

        # Log-likelihood of the joint copula
        joint_loglik = self.copula_xy['copula'].loglik(pseudo_joint)
        print(f"Contexted Shapes: {joint_data.shape}, {ctxt_joint.shape}, pseudo_joint: {pseudo_joint.shape}, {joint_loglik}")
        # Ensure output is a tensor of shape [num_points]
        if not isinstance(joint_loglik, torch.Tensor):
            joint_loglik = torch.tensor(joint_loglik, dtype=torch.float32)
        print(joint_loglik.std())
        return joint_loglik


    def _likelihood_x(self, X):
        """
        Compute the log-likelihood of the marginal copula c(X).

        Args:
        - X: Input data matrix of shape [num_points, d].

        Returns:
        - marginal_loglik: Log-likelihood of the marginal copula, shape [num_points].
        """
        ctxt_X = get_context(X.unsqueeze(0), self.correlations, self.p0_class)
        _, pseudo_X = evaluate_prcopula(X, ctxt_X, self.correlations, init_dist=self.p0_class)

        # Compute row-wise log-likelihood
        marginal_loglik = torch.tensor(
            [self.copula_x['copula'].loglik(pseudo_X[i:i+1]) for i in range(X.shape[0])],
            dtype=torch.float32
        )

        return marginal_loglik




    def _sample_marginal(self, num_samples):
        """
        Compute the log-probability of num_samples y values under the marginal density p_y(y).

        Args:
        - num_samples: Number of samples to retrieve from the marginal distribution.

        Returns:
        - log_p_y: Log-probability of sampled y values, shape [num_samples].
        """
        if self.p_y.shape[0] < num_samples:
            raise ValueError(f"Insufficient points in p_y. Expected at least {num_samples}, got {self.p_y.shape[0]}.")

        # Randomly sample num_samples points from p_y
        indices = torch.randperm(self.p_y.shape[0])[:num_samples]
        sampled_y_cdfs = self.p_y[indices]

        # Compute log-probabilities
        log_p_y = torch.log(sampled_y_cdfs + 1e-10)  # Avoid log(0) error
        return log_p_y




    def predict(self, X_input: pd.DataFrame, y_min: float = 0 + 1e-10, y_max: float = 1 - 1e-10, num_points: int = 20):
        """
        Predict outcomes for input data using the QB-Vine model.

        Args:
        - X_input: Input features (test set) as a DataFrame of shape [n_samples, d].
        - y_min: Minimum value of the y range for prediction.
        - y_max: Maximum value of the y range for prediction.
        - num_points: Number of grid points for y to evaluate.

        1. Get a range of possible y values (y_guesses).
        2. Sample marginal probability dist from trained y marginal.
        3. For Combine each sample with the y_guesses.
            - Creates a matrix of shape (len(y_guesses), dim) for each X sample.
            - We loop through the samples and create the matrix.
        4. For each sample and y_guess combination, compute:
            - Joint Likelyhood from fitted c(X, y)
            - Marginal Likelyhood from fitted c(X)
            - Likleyhood p(y|X) = p_y * c(X, y) / c(X)
        5. Each sample now has p(y_guess| sample) and we 
        6. Select the corresponding max likleyhood y for each sample.

        Returns:
        - predictions: Predicted y values for each row in X_input.
        """
        X = torch.tensor(X_input.to_numpy(), dtype=torch.float32)  # Shape: [n_samples, d]
        y_guesses = torch.linspace(y_min, y_max, num_points).unsqueeze(1)  # Shape: [num_points, 1]

        log_p_y = self._sample_marginal(num_samples = num_points)  # p_y Shape: [num_points]
        predictions = []

        for sample in X:
            x = sample.unsqueeze(0).repeat(num_points, 1)  # Shape: [num_points, d]
            joint_loglik = self._likelihood_joint(x=x, y=y_guesses)  # c(X, y) Shape: [num_points]
            marginal_loglik = self._likelihood_x(X=x)  # c(X) Shape: [num_points]
            print(f"Shapes: {joint_loglik}, {marginal_loglik}")
            print(f"STDs  : {joint_loglik.std()}, {marginal_loglik.std()}")
            
            conditional_loglik = joint_loglik + log_p_y - marginal_loglik
            best_y_index = torch.argmax(conditional_loglik)
            predictions.append(y_guesses[best_y_index].item())

        return torch.tensor(predictions, dtype=torch.float32)



    def save_model(self, folder_name: str):
        os.makedirs(folder_name, exist_ok=True)
        copulas = {'xy': self.copula_xy['copula'], 'x': self.copula_x['copula']}
        for name, copula in copulas.items():
            copula_path = os.path.join(folder_name, f'copula_{name}.json')
            copula.to_json(copula_path)
        model_data = {
            'init_dist': self.p0_class,
            'opt_corrs': self.correlations,
            'cdfs': getattr(self, 'cdfs', None),
            'copulas': list(copulas.keys()),
            'p_y': self.p_y.numpy() if isinstance(self.p_y, torch.Tensor) else self.p_y  # Save p_y
            }
        model_path = os.path.join(folder_name, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)


    @staticmethod
    def load_model(folder_dir: str):
        copulas = {}
        model_path = os.path.join(folder_dir, 'model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        for copula_name in model_data['copulas']:
            copula_path = os.path.join(folder_dir, f'copula_{copula_name}.json')
            copulas[copula_name] = pv.Vinecop(filename=copula_path)
        loaded_model = QBV(p0_class=model_data['init_dist'])
        loaded_model.copula_xy = {'copula': copulas.get('xy')}
        loaded_model.copula_x = {'copula': copulas.get('x')}
        loaded_model.correlations = model_data['opt_corrs']
        loaded_model.cdfs = model_data['cdfs']
        loaded_model.p_y = torch.tensor(model_data['p_y'], dtype=torch.float32)
    
        return loaded_model