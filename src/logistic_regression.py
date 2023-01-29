import numpy as np
import nevergrad as ng


class LogisticRegression:
    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

    def fit(self, X, y):
        n, d = X.shape
        self.n, self.d = n, d

        def _objective(params):
            w = params[:d].reshape(-1,1)
            b = params[d]

            z = X.dot(w) + b

            p = 1 / (1 + np.exp(-z))

            log_loss_component = -np.mean(y * np.log(p) + (1-y) * np.log(1 - p))
            l1_loss_component = self.l1 * np.sum(np.abs(w))
            l2_loss_component = self.l2 * np.sum(w ** 2)

            return log_loss_component + l1_loss_component + l2_loss_component

        optimizer = ng.optimizers.OnePlusOne(parametrization=d+1, budget=100)
        self.params = optimizer.minimize(_objective).value

    def predict_proba(self, X):
        w = self.params[:self.d].reshape(-1,1)
        b = self.params[self.d]
        z = X.dot(w) + b

        return 1 / (1 + np.exp(-z))
        
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
