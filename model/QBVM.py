import pyvinecopulib as pv
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.stats import norm


class QuasiBayesianVineRegression(BaseEstimator, RegressorMixin):
    def __init__(self, rho=0.5, bandwidth=1.0):
        self.rho = rho
        self.bandwidth = bandwidth
        self.marginal_predictives = {}
        self.vine_copula_model = None
        self.copula_buffer = None


    def _valid_input(self, X, y):
        return X.ndim == 2 and len(X) == len(y)


    def _initialize_marginals(self, X, y):
        """
        Initialize each marginal predictive distribution using a Normal prior.
        """
        for i in range(X.shape[1]):
            mean_init = np.mean(X[:, i])
            std_init = np.std(X[:, i])
            self.marginal_predictives[i] = norm(loc=mean_init, scale=std_init)
        self.marginal_predictives['y'] = norm(loc=np.mean(y), scale=np.std(y))


    def _update_marginal_predictive(self, prev_density, new_sample):
        """
        Update marginal predictive recursively using Bayesian recursion
        This updates the mean and variance based on the new data sample.
        """
        prev_mean, prev_std = prev_density.mean(), prev_density.std()

        # Bayesian update rule for mean and standard deviation.
        new_mean = prev_mean + self.rho * (new_sample - prev_mean)
        # TODO: Implement dynamic std. Consider how? Options?
        new_std = prev_std

        return norm(loc=new_mean, scale=new_std)


    def _fit_vine_copula(self, X, y):
        """
        Fit the vine copula model on the joint distribution of features and target.
        """
        _data = np.column_stack([X, y])
        pseudo_obs = self._pseudo_obs(_data)
        self.vine_copula_model = pv.Vinecop(pseudo_obs)
        # Store the pseudo-observations for future updates
        self.copula_buffer = pseudo_obs  


    def _update_vine_copula(self, new_X, new_y):
        """
        Update the vine copula with new data.
        """
        _new_data = np.column_stack([new_X, new_y])
        new_pseudo_obs = self._pseudo_obs(_new_data)
        
        # Update the copula buffer with new pseudo-observations
        self.copula_buffer = np.vstack([self.copula_buffer, new_pseudo_obs])
        self.vine_copula_model = pv.Vinecop(self.copula_buffer)


    def _pseudo_obs(self, data):
        """
        Compute the pseudo-observations (rank-transformed data scaled to [0, 1]).
        """
        ranks = np.argsort(np.argsort(data, axis=0), axis=0) + 1
        return ranks / (len(data) + 1)


    def _compute_conditional_distribution(self, X_row, _n_closest:int = 50):
        """
        Compute the conditional predictive distribution for the target given feature values.
        """
        marginals = []
        for i in range(len(X_row)):
            prev_density = self.marginal_predictives[i]
            updated_density = self._update_marginal_predictive(prev_density, X_row[i])
            self.marginal_predictives[i] = updated_density
            marginals.append(updated_density)

        if self.vine_copula_model:
            pseudo_obs_features = self._pseudo_obs(X_row.reshape(1, -1))

            # Simulate from the vine copula to predict the target.
            # Simulate full data, then condition on features
            joint_samples = self.vine_copula_model.simulate(10000)

            # Separate features and target from simulated data
            simulated_features = joint_samples[:, :-1]
            simulated_target = joint_samples[:, -1]

            # TODO: Improve this method. Is there a better way?
            # Find samples where the simulated features are closest to the actual feature values
            distance = np.linalg.norm(simulated_features - pseudo_obs_features, axis=1)
            closest_idx = np.argsort(distance)[:_n_closest]
            # Use the median target value of the closest samples as the prediction
            target_prediction = np.median(simulated_target[closest_idx])
        else:
            # If no copula model, return the product of marginal PDFs
            target_prediction = np.prod([m.pdf(X_row[i]) for i, m in enumerate(marginals)])

        return target_prediction


    def fit(self, X, y):
        """
        Fit the model by initializing and recursively updating the marginal predictives and vine copula.
        """
        X, y = np.asarray(X), np.asarray(y)
        if not self._valid_input(X, y):
            raise ValueError("Invalid input dimensions or data type")

        self._initialize_marginal_predictive_distribution(X, y)

        # Update the marginals recursively
        for i in range(X.shape[1]):
            for n in range(X.shape[0]):
                prev_density = self.marginal_predictives[i]
                current_density = self._update_marginal_predictive(prev_density, X[n, i])
                self.marginal_predictives[i] = current_density

        for n in range(len(y)):
            prev_density_y = self.marginal_predictives['y']
            current_density_y = self._update_marginal_predictive(prev_density_y, y[n])
            self.marginal_predictives['y'] = current_density_y

        # Fit the vine copula on the joint data (features + target)
        self._fit_vine_copula(X, y)

        self.fitted_ = True
        return self


    def predict(self, X):
        if not hasattr(self, 'fitted_'):
            raise ValueError("This QuasiBayesianVineRegression instance is not fitted yet.")

        X = np.asarray(X)
        predictions = []

        for n in range(X.shape[0]):
            joint_predictive = self._compute_conditional_predictive(X[n, :])
            predictions.append(joint_predictive)

        return np.array(predictions)


    def update(self, new_X, new_y):
        """
        Update the model with new data points (recursive update for marginals and copula).
        """
        for i in range(new_X.shape[1]):
            for n in range(new_X.shape[0]):
                prev_density = self.marginal_predictives[i]
                current_density = self._update_marginal_predictive(prev_density, new_X[n, i])
                self.marginal_predictives[i] = current_density

        for n in range(len(new_y)):
            prev_density_y = self.marginal_predictives['y']
            current_density_y = self._update_marginal_predictive(prev_density_y, new_y[n])
            self.marginal_predictives['y'] = current_density_y

        # Update the vine copula with the new data
        self._update_vine_copula(new_X, new_y)