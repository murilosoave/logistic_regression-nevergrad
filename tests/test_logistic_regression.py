import numpy as np
import pytest

from src.logistic_regression import LogisticRegression


@pytest.fixture
def toy_dataset():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    return X, y


def test_fit(toy_dataset):
    X, y = toy_dataset

    # Initialize the LogisticRegression model
    model = LogisticRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Check if the fit method returns expected results
    assert isinstance(model.params, np.ndarray), "Expected the params to be a numpy ndarray."
    assert model.params.shape == (3 + 1,), "Expected the shape of params to be (3 + 1,)."


def test_predict_proba(toy_dataset):
    X, y = toy_dataset

    # Initialize the LogisticRegression model
    model = LogisticRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Predict the probabilities for X
    proba = model.predict_proba(X)

    # Check if the predict_proba method returns expected results
    assert isinstance(proba, np.ndarray), "Expected the proba to be a numpy ndarray."
    assert proba.shape == (3,), "Expected the shape of proba to be (3,)."