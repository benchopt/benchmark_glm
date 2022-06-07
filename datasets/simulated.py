from benchopt import BaseDataset
from benchopt.datasets import make_correlated_data


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

        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, rho=self.rho,
            random_state=self.random_state
        )

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(X=X, y=y > 0)
