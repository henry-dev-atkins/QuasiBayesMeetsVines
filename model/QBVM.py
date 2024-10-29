import pyvinecopulib as pv
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.stats import gaussian_kde, norm, cauchy, uniform


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

        Purpose: 
            This copula models dependencies between two variables in the recursive density update.
        Implementation: 
            This bivariate copula acts as a basic unit for pairwise dependencies and needs to be 
            independently implemented for updating univariate densities. 
        Why Seperate Copulas?
            The code must allow for flexible bivariate copula selection and use conditional 
            dependency values as per Equation 2, which cannot directly reuse the full vine 
            structure or KDE components.
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

        Purpose: 
            Equation (4) involves a conditional copula transformation specifically used for 
            recursively updating the cumulative distribution function. It provides updates to 
            marginal distributions, transforming the cumulative distributions by conditioning 
            on observed data.
        Implementation: 
            This conditional copula, associated with Equation (4), requires a conditional 
            structure (i.e., Hρ as in Equation (5)), involving specialized calculations that 
            differ from simple bivariate or vine copulas. 
        Why Seperate Copulas?
            This implementation should ideally not be combined with the vine copula but instead 
            remain focused on conditioning for cumulative distributions.
        """
        print(cdf_x)
        print(cdf_y)
        data = np.vstack((cdf_x, cdf_y)).T
        bicop_model = pv.Bicop()
        bicop_model.select(data=data)
        
        # Conditional CDF H(cdf_x | cdf_y) with first h-function
        H = bicop_model.hfunc1(data)
        return H
    

    def _update_dist(self, al, p_x, p_xn):
        """
        Equation 4 from Paper.
        """
        _term1 = (1 - al) * p_x
        term_2 = al * self.conditional_copula(p_x, p_xn)
        return _term1 + term_2


    def initialise_recursions(self, N: int = 2, distribution_type: str = 'uniform'):
        """
        Initializes with either a Uniform or Cauchy Distribution.

        Parameters:
        - N: int, number of samples to sim.
        - distribution_type: str, either 'uniform' or 'cauchy' to choose the initialization type.

        Returns:
        - data: dict, initialized data with specified distribution type.
        """
        if distribution_type == 'cauchy':
            #TODO: Check range
            x_values = np.linspace(-10, 10, N)
            data = {
                'x_density': cauchy.pdf(x_values),
                'x_distribution': cauchy.cdf(x_values),
                'x_distribution_n': cauchy.cdf(x_values),
                'y_distribution': cauchy.cdf(x_values),
            }
        elif distribution_type == 'uniform':
            data = {
                'x_density': np.ones(N),
                'x_distribution': np.sort(np.random.uniform(0, 1, N)),
                'x_distribution_n': np.sort(np.random.uniform(0, 1, N)),
                'y_distribution': np.sort(np.random.uniform(0, 1, N))
                }
            for key, value in data.items():
                print(key, value.shape)
        else:
            raise ValueError(f"initialise_recursions: Distribution_type must be 'uniform' or 'cauchy', but is {distribution_type}.")

        return data

    def recurse_samples(self, alpha, samples):
        """
        Applies sample-wise recursion to update the distribution and density functions.
        """
        _iter_data = self.initialise_recursions()

        for sample in samples:
            # density p(x) (Equation 3)
            sample = np.array([sample])
            print("Samples:", _iter_data['x_distribution'].shape, sample.shape)
            _iter_data['x_density'] *= (1 - alpha) + (alpha * self.bivariate_copula(_iter_data['x_distribution'], sample))

            # Cumulative Distribution P(x) (Equation 4)
            _iter_data['x_distribution_last_sample'] = _iter_data['x_distribution'] #Store for next sample.
            _iter_data['x_distribution'] = self._update_dist(alpha, sample, _iter_data['x_distribution'])

            # y_density p(y | x) (Equation 12)
            _iter_data['y_density'] = self._update_dist(alpha, _iter_data['y_distribution'], sample)

        return _iter_data


    def get_marginals(self, data: np.ndarray):
        """
        Get Marginal Densities and Distributions.
        Iterate Equations (3) and (4) from paper.

        Input:
            - data: The dataset with shape (n_samples, n_features)

        Output:
            - marginals: dict, containing the marginal distributions and densities for each dimension.
        """
        dimensions = data.shape[1]
        marginals = {}

        for n in range(0, dimensions):
            alpha_n = (2 - (n^-1)) * (1 / (1+n))
            marginals[n] = self.recurse_samples(alpha_n, samples=data[:, n])
        return marginals


    def estimate_copula(self, marg_dists: np.ndarray):
        """
        Equation 10.
        Get Copula (used for both the Joint Copula and X Copula).
        
        Input:
            - Marginal Distributions (P_n)
        Output:
            - c: Copula 
        
        Purpose: 
            The final vine copula in Equations (10) and (11) models the full multivariate 
            dependency structure, integrating all dimensions via a vine structure composed 
            of several bivariate copulas in a tree-based hierarchy.
        Implementation: 
            The vine copula structure is distinct because it builds upon multiple layers 
            of pairwise copulas with a hierarchy, enabling high-dimensional dependency modeling. 
        Why Seperate Copulas?    
            This requires a dedicated structure that aggregates multiple conditional copulas 
            across trees, which is computationally intensive and designed separately from the 
            simpler updates used in Equations (2) and (4).
        """
        copula_model = pv.Vinecop(data=marg_dists)

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
        # NOTE: Not in Use yet.
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
        _recursive_data = np.hstack((X, y))
        marginals = self.get_marginals(_recursive_data)
        final_marginals = marginals[X.shape[1]]

        # Estimate Copula for Combining the Marginals.
        joint_copula = self.estimate_copula(np.hstack((final_marginals['x_distribution'], final_marginals['y_distribution'])))
        x_copula = self.estimate_copula(final_marginals['x_distribution'])

        #TODO: Check gaussian_kde is correct here
        # Use a scipy Gaussian KDE to represent the marginal distribution of y.
        self.model = {
            'joint_copula': joint_copula,
            'x_copula': x_copula,
            'y_marginal': final_marginals['y_distribution'],
            'y_max': y.max(),
            'y_min': y.min()
            }
        return self.model
    

    def predict_sample(self, _X, _iters:int=1000):
        """
        Equation 12.
        Predict the probability density for y given single sample input X.
        
        Parameters:
        -----------
        X : array-like
            Input vector for which to predict y.
        
        Returns:
        --------
        y_density : float
            Predicted density for y given X.
        """
        c_X = self.model['x_copula'].pdf(_X)

        y_predictions = np.zeros(_iters)
        y_range = np.linspace(self.model['y_min'], self.model['y_max'], _iters)
        for y in y_range:
            joint_vars = np.hstack(([y], _X))
            c_y_X = self.model['joint_copula'].pdf(joint_vars)
            p_y = self.model['y_marginal'].pdf(y)
            conditional_density = (c_y_X * p_y) / c_X
            y_predictions[int(y)] = conditional_density.mean()
        
        # TODO: Consider alternatives to mean:
        #       - Weighted Mean Based on Conditional Density
        #       - Mode of the Conditional Distribution
        #       - Quantile-based Mean (Trimmed Mean)
        #       - Expected Value Calculation Using Trapezoidal Integration
        return y_predictions.mean()
    

    def predict(self, X:np.ndarray):
        self.predictions = np.zeros(X.shape[0])
        if len(X.shape) == 1:
            self.predictions = self.predict_sample(X)
            return self.predictions
        for i, sample in enumerate(X):
            self.predictions[i] = self.predict_sample(sample)
        return self.predictions


if __name__ == '__main__':
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import MinMaxScaler

    feat, target = load_wine(return_X_y=True)

    feat = MinMaxScaler().fit_transform(feat)
    target = MinMaxScaler().fit_transform(target.reshape(-1, 1))

    model = QuasiBayesianVineRegression()
    model.fit(feat, target[:len(feat)])

    model.predict(feat[2])