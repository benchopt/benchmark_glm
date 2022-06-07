import numpy as np


from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import Ridge, TweedieRegressor


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'scikit-learn'

    install_cmd = "conda"
    requirements = [
        'pip:git+https://github.com/lorentzenchr'
        '/scikit-learn#glm_newton_cholesky'
    ]

    # any parameter defined here is accessible as a class attribute
    parameters = {'solver': ['auto', 'cholesky']}

    def set_objective(self, X, y, datafit, penalty, reg):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X, self.y = X, y

        if datafit == "l2":
            if penalty == "l2":
                self.clf = Ridge(
                    alpha=reg, fit_intercept=False, solver=self.solver
                )
            else:
                raise NotImplementedError()
        elif datafit == "tweedie":
            # TODO not sure what is happenning with penalty?
            self.clf = TweedieRegressor(fit_intercept=False)
        else:
            raise NotImplementedError()

    def run(self, n_iter):
        if n_iter == 0:
            self.coef_ = np.zeros(self.X.shape[1])
            return

        self.clf.set_params(max_iter=n_iter)
        self.clf.fit(self.X, self.y)
        self.coef_ = self.clf.coef_

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.coef_
