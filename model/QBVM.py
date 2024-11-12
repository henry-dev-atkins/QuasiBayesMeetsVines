
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

from .utils import cGC_distribution, minmax_unif, grids_cdfs, linear_energy_grid_search, get_context, evaluate_prcopula, cdf_std_normal, inverse_std_normal, cdf_std_normal

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
        train_data, test_data = train_test_split(data, train_size=self.train_frac, random_state=self.seed)
        y_train = (train_data - train_data.mean()) / train_data.std()
        self.mean_y, self.std_y = train_data.mean(), train_data.std()
        y_test = (test_data - self.mean_y) / self.std_y
        return torch.tensor(y_train.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)


    def fit(self, data):
        y_train, y_test = self._preprocess_data(data)
        y_permutations = torch.stack(Parallel(n_jobs=-1)(delayed(lambda: y_train[torch.randperm(y_train.shape[0])])() for _ in range(self.perm_count)))
        scores_dic = self._optimize_theta(y_permutations)
        optimum_thetas = self._extract_optimal_thetas(scores_dic)
        self.context = self._build_context(y_permutations, optimum_thetas)
        self.model_params = self._fit_copulas(y_train, y_test, optimum_thetas)


    def predict(self, test_data):
        y_test = (test_data - self.mean_y) / self.std_y
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        test_dens, cop_dens_xy = evaluate_prcopula(y_test_tensor, self.context, self.model_params['optimum'], init_dist=self.init_dist)
        return test_dens, cop_dens_xy

    def _optimize_theta(self, y_permutations):
        size = 50
        theta_grids = torch.linspace(0.1, 0.99, size)
        scores_dic = torch.zeros([size, y_permutations.shape[2]])
        scores_dic = torch.stack(Parallel(n_jobs=-1)(delayed(linear_energy_grid_search)
                                (y_permutations, torch.full(
                                    (y_permutations.shape[2],),
                                        theta_grids[grids]
                                    ), 
                                    beta=0.5, 
                                    init_dist=self.init_dist) for grids in range(size)
                                    )
                                )
        
        return scores_dic

    def _extract_optimal_thetas(self, scores_dic):
        return torch.argmin(scores_dic, dim=0)

    def _build_context(self, y_permutations, optimum_thetas):
        return get_context(y_permutations, optimum_thetas, init_dist=self.init_dist)

    def _fit_copulas(self, y_train, y_test, optimum_thetas):
        optband_xy = 3.0
        controls_xy = pv.FitControlsVinecop(
                                            family_set=[pv.BicopFamily.tll],
                                            selection_criterion='mbic',
                                            nonparametric_method='constant',
                                            nonparametric_mult=optband_xy
                                            )
        cop_xy = pv.Vinecop(y_train, controls=controls_xy)
        return {'cop_xy': cop_xy, 'optimum': optimum_thetas}

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)