"""
Created by Federico Amato
2022 - 08 - 09

Create a Gaussian Regression model from SciKit Learn.
"""
import joblib  # save and load models with dump and load methods
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

import sys
sys.path.append('src')
# from data.preprocess import get_features, get_target

def get_features(data):
    features = data.drop(labels='ETa', axis=1)
    return features


def get_target(data):
    target = data['ETa']
    return target

class GaussianRegressionModel(GaussianProcessRegressor):
    def __init__(self, data, seed=1998, **params):
        # Initialize sklearn model
        super().__init__(**params, random_state=seed)
        self.seed = seed
        # Take features and target from data
        self.X = get_features(data)
        self.y = get_target(data)
        # Create train and test splits
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=self.seed
        )

    def train(self, X_train=None, y_train=None):
        """
        Take train data from input or from initial configuration
        and call the parent fit method with those.
        """
        # Check if train data comes from method input
        if X_train is not None:
            if y_train is not None:
                X = X_train
                y = y_train
            else:
                print("Please specify train target")
        elif y_train is not None:
            print("Please specify train features")
        # Or take train data from class configuration
        else:
            X = self.X_train
            y = self.y_train
        # Fit on data
        self.fit(X, y)
        return None

    def predict(self, X_predict=None):
        if X_predict is not None:
            X = X_predict
        else:
            X = self.X_test
        # Predict data
        y_predict = super().predict(X)
        return y_predict

    def test(self, X_test=None, y_test=None):
        if X_test is not None:
            if y_test is not None:
                X = X_test
                y = y_test
            else:
                print("Please specify test target")
        elif y_test is not None:
            print("Please specify test features")
        else:
            X = self.X_test
            y = self.y_test
        # Take coefficient of determination of the prediction
        score = self.score(X, y)
        return score

    def save(self, fname='gaussian_process.joblib'):
        fname = PROJ_ROOT + '/eta_ml/models/' + fname
        dump(self, fname)

