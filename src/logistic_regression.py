import numpy as np
import nevergrad as ng


class LogisticRegression:
    """
    Logistic Regression Classifier with L1 and L2 Regularization.

    Attributes
    ----------
    l1 : float, optional (default=0)
        The weight of the L1 regularization term.
    l2 : float, optional (default=0)
        The weight of the L2 regularization term.
    n : int
        The number of training samples (set in the fit method).
    d : int
        The number of features in the training data (set in the fit method).
    params : ndarray
        The optimized model parameters (set in the fit method).

    Methods
    -------
    fit(X, y)
        Fit the logistic regression model to the training data (X, y).
    predict_proba(X)
        Return the predicted probabilities for a given input X.
    predict(X)
        Return the class predictions (0 or 1) based on the predicted probabilities.
    """
    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data (X, y).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input training data.
        y : ndarray, shape (n_samples,)
            The target labels, where y[i] in (0,1) .

        Returns
        -------
        self : instance
            Self object.
        """
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
        
        return self

        optimizer = ng.optimizers.OnePlusOne(parametrization=d+1, budget=100)
        self.params = optimizer.minimize(_objective).value

    def predict_proba(self, X):
        """
        Predict the probabilities of the positive class for a given input X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input data to predict probabilities for.

        Returns
        -------
        proba : ndarray, shape (n_samples,)
            The predicted probabilities of the positive class.
        """
        w = self.params[:self.d].reshape(-1,1)
        b = self.params[self.d]
        z = X.dot(w) + b

        return 1 / (1 + np.exp(-z))
        
    def predict(self, X):
        """
        Predict the class labels (0 or 1) for a given input X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input data to predict class labels for.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            The predicted class labels, where 0 <= labels[i] <= 1.
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
