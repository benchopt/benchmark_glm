import numpy as np


from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    from sklearn.linear_model import LogisticRegression

    from sklearn._loss import HalfBinomialLoss
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
        'scikit-learn@glm_newton_cholesky'
    ]

    # any parameter defined here is accessible as a class attribute
    parameters = {'solver': [
        'lbfgs2', 'lbfgs', 'newton-cg', 'newton-cholesky'
    ]}

    stopping_criterion = SufficientProgressCriterion(
        patience=5, strategy='iteration'
    )

    def set_objective(self, X, y, w, reg):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X, self.y, self.w = X, y, w

        if self.solver in ['lbfgs', 'newton-cg']:
            self.clf = LogisticRegression(
                C=2 / reg / X.shape[0], solver=self.solver, tol=1e-16,
                fit_intercept=False
            )
        else:
            solver = self.solver.replace('2', '')
            self.clf = BinomialRegressor(
                solver=solver, alpha=reg, tol=1e-16, max_iter=1,
                fit_intercept=False
            )

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
        return self.coef_.flatten()
