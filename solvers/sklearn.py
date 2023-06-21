import warnings

import numpy as np


from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:

    from sklearn.exceptions import ConvergenceWarning

    from sklearn._loss import HalfBinomialLoss
    from sklearn.linear_model import PoissonRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model._glm.glm import _GeneralizedLinearRegressor

    class BinomialRegressor(_GeneralizedLinearRegressor):
        def _get_loss(self):
            return HalfBinomialLoss()


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'scikit-learn'

    install_cmd = "conda"
    requirements = [
        'pip:git+https://github.com/lorentzenchr/'
        'scikit-learn@glm_newton_lsmr_only'
    ]

    # any parameter defined here is accessible as a class attribute
    parameters = {'solver': [
        'newton-lsmr', 'lbfgs', 'newton-cg', 'newton-cholesky'
    ]}

    stopping_criterion = SufficientProgressCriterion(
        patience=3, strategy='iteration'
    )

    def get_next(self, stop_val):
        return int(max(stop_val + 1, stop_val * 1.3))

    def skip(self, X, y, w, datafit, reg, fit_intercept=False):
        if datafit == "poisson" and self.solver in ["newton-cg"]:
            return True, "solvers only compared for binom datafit"
        return False, None

    def set_objective(self, X, y, w, datafit, reg, fit_intercept=False):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X, self.y, self.w = X, y, w
        self.fit_intercept = fit_intercept

        if datafit == "binom":
            self.clf = LogisticRegression(
                C=2 / reg / X.shape[0], solver=self.solver, tol=1e-16,
                fit_intercept=fit_intercept
            )
        else:
            self.clf = PoissonRegressor(
                solver=self.solver, alpha=reg, tol=1e-16, max_iter=1,
                fit_intercept=fit_intercept
            )

    def run(self, n_iter):
        if n_iter == 0:
            self.coef_ = np.zeros(self.X.shape[1] + self.fit_intercept)
            return

        self.clf.set_params(max_iter=n_iter)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.clf.fit(self.X, self.y)
        self.coef_ = self.clf.coef_.flatten()

        if self.fit_intercept:
            self.coef_ = np.r_[self.coef_, self.clf.intercept_]

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.coef_
