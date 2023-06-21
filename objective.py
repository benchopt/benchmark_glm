from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from sklearn._loss import HalfPoissonLoss
    from sklearn._loss import HalfBinomialLoss
    from sklearn.linear_model._linear_loss import LinearModelLoss

    lml = LinearModelLoss(base_loss=HalfBinomialLoss(), fit_intercept=True)


class Objective(BaseObjective):
    name = "Generalized Linear Model"

    # use sklearn modified scikit-learn to compute the loss functions
    # and test extra solvers.
    install_cmd = "conda"
    requirements = [
        'pip:git+https://github.com/lorentzenchr/'
        'scikit-learn@glm_newton_lsmr_only'
    ]

    parameters = {
        'datafit': ['poisson', 'binom'],
        'reg': [1e-4, 1e-12]
    }

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def set_data(self, X_train, y_train, w_train, X_test, y_test, w_test):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.X_train, self.y_train, self.w_train = X_train, y_train, w_train
        self.X_test, self.y_test, self.w_test = X_test, y_test, w_test

        if self.datafit == "binom":
            y_thresh = np.quantile(y_train, q=0.95)
            self.y_train = (y_train > y_thresh).astype(np.float64)
            self.y_test = (y_test > y_thresh).astype(np.float64)

            self.lml = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=True
            )
        elif self.datafit == "poisson":
            self.lml = LinearModelLoss(
                base_loss=HalfPoissonLoss(), fit_intercept=True
            )

    def compute(self, beta):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        train_loss = self.lml.loss(
            coef=beta,
            X=self.X_train, y=self.y_train.astype(np.float64),
            sample_weight=self.w_train, l2_reg_strength=self.reg,
        )
        test_loss = self.lml.loss(
            coef=beta,
            X=self.X_test, y=self.y_test.astype(np.float64),
            sample_weight=self.w_test,
            l2_reg_strength=self.reg,
        )
        return dict(value=train_loss, test_loss=test_loss)

    def get_objective(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(
            X=self.X_train, y=self.y_train, w=self.w_train,
            datafit=self.datafit, reg=self.reg,
            fit_intercept=self.fit_intercept
        )
