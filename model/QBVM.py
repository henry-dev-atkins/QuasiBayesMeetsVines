import pyvinecopulib as pv
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.stats import gaussian_kde, norm


class QuasiBayesianVineRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = {}


    def bivariate_copula(self, cdf_x: np.ndarray, cdf_y: np.ndarray) -> float:
        """
        Automatically selects the best-fitting bivariate copula family and computes the copula density.
        
        Parameters:
        - cdf_x: CDF values for the first variable (array-like)
        - cdf_y: CDF values for the second variable (array-like)
        
        Returns:
        - copula_density: The copula density value for the selected copula family.
        """
        data = np.vstack((cdf_x, cdf_y)).T
        bicop_model = pv.Bicop()
        bicop_model.select(data=data)
                
        return bicop_model.pdf(data)


    def conditional_copula(self, cdf_x: np.ndarray, cdf_y: np.ndarray) -> np.ndarray:
        """
        Computes H(cdf_x, cdf_y), the conditional copula-based function using the best-fit copula model.
        Automatically selects the copula family using Bicop.select().

        Parameters:
        - cdf_x: CDF values for the first variable (array-like).
        - cdf_y: CDF values for the second variable (array-like).

        Returns:
        - H: The conditional CDF value H(cdf_x | cdf_y) based on the selected copula family.
        """
        data = np.vstack((cdf_x, cdf_y)).T
        bicop_model = pv.Bicop()
        bicop_model.select(data=data)
        
        # Conditional CDF H(cdf_x | cdf_y) with first h-function
        H = bicop_model.hfunc1(data)
        return H
    

    def _update_dist(self, al, _dist, _n_dists):
        """
        Equation 4 from Paper.
        """
        _term1 = (1 - al) * _dist
        term_2 = al * self.conditional_copula(_dist, _n_dists)
        return _term1 + term_2
    

    def apply_recursion(self, alpha: float, last_dist: np.ndarray, last_n_dists: np.ndarray, last_density: np.ndarray, last_y_dist: np.ndarray):
        _iter_data = {}

        # Equation 3: Update p(x)
        _iter_data['x_density'] = last_density * ((1 - alpha) + (alpha * self.bivariate_copula(last_dist, last_n_dists)))

        # Equation 4: Update for P(x)
        _iter_data['x_distribution'] = self._update_dist(alpha, last_dist, last_n_dists)

        # Update P(x^n) for the next iteration (future use)
        # TODO: Validate that the order of inputs is correct - gpt told me to do this!
        _iter_data['x_distribution_n'] = self._update_dist(alpha, last_n_dists, last_dist)

        # Equation 12: Update p(y|x) 
        _iter_data['y_distribution'] = self._update_dist(alpha, last_y_dist, last_n_dists)

        return _iter_data
    
    def initialise_recursions(self, ns:int):
        """
        Currently applies Uniform Dist.
        """
        data = {
                'x_density': np.ones(ns),
                'x_distribution': np.zeros(ns),
                'x_distribution_n': np.zeros(ns),
                'y_distribution': np.zeros(ns)
            }

        # Normalize arrays to [0, 1] range
        data['x_distribution'] = np.linspace(0, 1, ns)
        data['x_distribution_n'] = np.linspace(0, 1, ns)
        data['y_distribution'] = np.linspace(0, 1, ns)

        # Calculate density
        density = 1 / ns

        # Update x_density with the calculated density
        data['x_density'] = np.full(ns, density)
        return data


    def recurse_marginals(self, data: np.ndarray):
        """
        Get Marginal Densities and Distributions.
        Iterate Equations (3) and (4) from paper.

        Input:
            - data: The dataset with shape (n_samples, n_features)

        Output:
            - marginals_x: A dictionary containing updated marginal densities and distributions
        """
        ns = data.shape[0]  # number of datapoints/observations.
        marginals = {0: self.initialise_recursions(ns)}

        for n in range(1, ns):
            last_dist = marginals[n-1]['x_distribution']
            last_n_dists = marginals[n-1]['x_distribution_n']
            last_density = marginals[n-1]['x_density']
            last_y_dist = marginals[n-1]['y_distribution']
            alpha_n = (2 - (n^-1)) * (1 / (1+n))

            marginals[n] = self.apply_recursion(alpha_n, last_dist, last_n_dists, last_density, last_y_dist)

        return marginals


    def estimate_copula(self, marg_dists: np.ndarray):
        """
        Equation 10.
        Get Copula (used for both the Joint Copula and X Copula).
        
        Input:
            - Marginal Distributions (P_n)
        Output:
            - c: Copula 
        """
        copula_model = pv.Vinecop(data=marg_dists)

        copula_density = copula_model.pdf(marg_dists)

        return copula_model


    def compute_pairwise_copula(self, dist_i, dist_j):
        """
        Compute the pairwise KDE-based copula c_ij between two marginal distributions
        using Equation (11) from the paper.

        Input:
            dist_i: Marginal distribution P_i
            dist_j: Marginal distribution P_j
        Output:
            c_ij: Pairwise copula value between P_i and P_j
        """
        # TODO: Implement Actual KDE-based copula estimation here (Equation 11)
        # THIS EQUATION IS A PLACEHOLDER!
        kde_copula = np.exp(-0.5 * (np.linalg.norm(dist_i - dist_j))**2)
        return kde_copula


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Quasi-Bayesian Vine Model on the input data.
        
        Parameters:
        X: np.ndarray
            Input feature data with shape (n_samples, n_features)
        y: np.ndarray
            Target values with shape (n_samples,)

        Returns:
        dict
            A dictionary with the fitted marginals and copula model.
        """

        # Recursive Marginals
        _recursive_data = np.hstack((X, y.reshape(-1, 1)))
        marginals = self.recurse_marginals(_recursive_data)
        final_marginals = marginals[X.shape[1]]

        # Estimate Copula for Combining the Marginals.
        joint_copula = self.estimate_copula(np.hstack((final_marginals['x_distribution'], final_marginals['y_distribution'])))
        x_copula = self.estimate_copula(final_marginals['x_distribution'])

        #TODO: Check gaussian_kde is correct here
        # Use a scipy Gaussian KDE to represent the marginal distribution of y.
        self.model = {
            'joint_copula': joint_copula,
            'x_copula': x_copula,
            'y_marginal': gaussian_kde(final_marginals['y_distribution']),
            'y_max': y.max(),
            'y_min': y.min()
            }
        return self.model
    

    def predict_sample(self, _X, _iters:int=1000):
        """
        Equation 12.
        Predict the probability density for y given input X.
        
        Parameters:
        -----------
        X : array-like
            Input vector for which to predict y.
        
        Returns:
        --------
        y_density : float
            Predicted density for y given X.
        """
        print(_X)
        c_X = self.model['x_copula'].pdf(_X)

        y_predictions = np.zeros(_iters)
        y_range = np.linspace(self.model['y_min'], self.model['y_max'], _iters)
        for y in y_range:
            joint_vars = np.hstack(([y], _X))
            c_y_X = self.model['joint_copula'].pdf(joint_vars)
            p_y = self.model['y_marginal'].pdf(y)
            conditional_density = (c_y_X * p_y) / c_X
            y_predictions[int(y)] = conditional_density.mean()
        
        return y_predictions.mean()
    

    def predict(self, X:np.ndarray):
        self.predictions = np.zeros(X.shape[0])
        if len(X.shape) == 1:
            self.predictions = self.predict_sample(X)
            return self.predictions
        for i, sample in enumerate(X):
            self.predictions[i] = self.predict_sample(sample)
        return self.predictions