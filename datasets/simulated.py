from benchopt import BaseDataset
from benchopt import safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200)
        ],
        'rho': [0, 0.6],
        'random_state': [27]
    }

    def get_data(self):

        rng = np.random.RandomState(self.random_state)

        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, rho=self.rho,
            random_state=rng
        )

        # Generate poisson variables
        mu = np.exp(y)
        y = rng.poisson(mu).astype(np.float64)

        w = np.full_like(y, fill_value=(1.0 / y.shape[0]))

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, w, random_state=0
        )

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(
            X_train=X_train, y_train=y_train, w_train=w_train,
            X_test=X_test, y_test=y_test, w_test=w_test
        )
