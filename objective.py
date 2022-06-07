from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Generalized Linear Model"

    parameters = {
        'datafit': ['l2'],
        'penalty': ['l2'],
        'reg': [0.1]
    }

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def skip(self, X, y):
        if self.datafit == 'logreg' and len(np.unique(y)) > 2:
            return True, "y is not for binary classif"
        return False, None

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.X, self.y = X, y

    def compute(self, beta):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        if self.datafit != 'logistic' and self.penalty != 'l2':
            raise NotImplementedError("TODO implement other loss ogrisel")
        diff = self.y - self.X.dot(beta)
        return .5 * diff.dot(diff) + 0.5 * self.reg * (beta @ beta)

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(X=self.X, y=self.y, datafit=self.datafit,
                    penalty=self.penalty, reg=self.reg)
