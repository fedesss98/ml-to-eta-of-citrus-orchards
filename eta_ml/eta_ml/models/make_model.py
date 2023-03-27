# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 22:00:00 2022

@author: Federico Amato

Train a model and test it.
Validation on 2022 measures.
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load  # To save model

import logging

# from .. neptune.log_neptune import log_neptune

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent


class ModelTrainer:
    def __init__(self, model, model_name, features, visualize_error=False, **kwargs):
        logging.info(f'\n\n{"-"*7} {model_name.upper()} MODEL TRAINING {"-"*7}\n\n')
        
        self.features = features
        self.model = model
        self.model_name = model_name
        # Relative Error DataFrame
        self.kt = pd.DataFrame()
        # Find folds data
        k = len(list(ROOT_DIR.glob('data/processed/test_fold_*')))
        # Initialize models dictionary
        models = dict()
        self.scores = np.zeros((k, 2))
        for fold in range(k):
            trained_model = self.train_model(self.model, fold)
            models[fold] = trained_model
            self.scores[fold] = self.test_model(trained_model, fold)
        if visualize_error:
            self.print_error()
        self.log_model_scores()
        logging.info(f'R2 Scores Mean: {self.scores.mean(axis=0)[0]:.2f}')
        logging.info(f'R2 Scores Max: {self.scores.max(axis=0)[0]:.2f}')
        logging.info(f'Score Variance: {self.scores.var(axis=0)[0]:.4f}')
        # Save the best scoring model
        self.best_model = models[np.argmax(self.scores, axis=0)[0]]
        self.save_model()
        # if log:
        #     log_run(model, np.array(scores), **kwargs)
        logging.info(f'\n\n{"/"*30}\n')
        return None

    @staticmethod
    def error_function(x):
        """
        Set 1
        a = .5
        phase = -.5
        q = 0.2
        d = 10
        v = 60 * 1/(x+d)
        """
        # Parameters
        a = .5
        phase = -.5
        q = 0.2
        d = 10
        v = 60 * 1/(x+d)
        # Shifted Sine function
        y = a*np.sin(np.pi*v*x - np.pi*phase) - q
        return y

    def train_model(self, model, k):
        train = pd.read_pickle(ROOT_DIR/'data/processed'/f'train_fold_{k}.pickle')
        X_train = train.loc[:, self.features]
        y_train = train.loc[:, 'ETa'].values.ravel()
        model.fit(X_train, y_train)
        return model

    def print_error(self):
        # Load complete set of measures
        measures = pd.read_pickle(
            ROOT_DIR / 'data/processed'/'processed.pickle')['ETa']
        measures = measures.drop(index=pd.date_range('2018-01-01','2019-01-01'))
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 10))
        fig.suptitle("Relative Errors of predictions across folds")
        ax_ref = ax2.twinx()
        ax_interp = ax2.twiny()
        g1 = sns.scatterplot(self.kt,
                             x = 'Day',
                             y = 'Kt',
                             hue = 'fold',
                             ax = ax1)
        g2 = sns.scatterplot(self.kt,
                             x = 'Day',
                             y = 'Kt',
                             hue = 'fold',
                             ax = ax2)
        measures.plot(ax=ax_ref, color='orange', zorder=0, alpha=0.3)
        ax_ref.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(-1, 1)
        ax2.set_title("Zoom in w/ respect to ETa scaled")
        ax2.set_zorder(2)
        ax2.set_facecolor('none')
        x_interp = np.linspace(0, 1, 200)
        y_interp = self.error_function(x_interp)
        ax_interp.plot(x_interp, y_interp)
        plt.tight_layout()
        plt.show()
        return None

    def make_kt(self, kt: pd.Series, k: int):
        # Remove outliers farther from 2 times the std
        kt.loc[kt.abs() > 6*kt.std()] = np.nan
        # Make it a DataFrame
        kt = kt.to_frame(name='Kt')
        # Add fold information
        kt['fold'] = k + 1
        self.kt = pd.concat([self.kt, kt])

    def test_model(self, model, k):
        test = pd.read_pickle(ROOT_DIR/'data/processed'/f'test_fold_{k}.pickle')
        X_test, y_test = test.loc[:, self.features], test.loc[:, 'ETa']
        prediction_test = model.predict(X_test)
        # Compute scores on scaled values
        r2_scaled = r2_score(y_test, prediction_test)
        rmse_scaled = mean_squared_error(y_test, prediction_test, squared=False)
        # Rescale values
        prediction_test = self.rescale_series(prediction_test)
        measures_test = self.rescale_series(y_test.values, y_test.index).squeeze()
        # Compute scores on rescaled values
        r2 = r2_score(measures_test, prediction_test)
        rmse = mean_squared_error(measures_test, prediction_test, squared=False)
        kt = prediction_test / measures_test - 1
        self.make_kt(kt, k)
        logging.info(f'R2 score on test {k}: {r2_scaled:.2f} - {r2:.2f}'
                     f'\nRMSE score on test: {rmse_scaled:.2f} - {rmse:.2f}')
        scores = (r2, rmse)
        return scores

    def save_model(self):
        dump(self.best_model, ROOT_DIR/'models'/f'{self.model_name}.joblib')
        return None

    def log_run(self, model, size, scores):
        logging.info(model)
        logging.info(f'Scores Mean: {scores.mean():.4f}')
        logging.info(f'Score Variance: {scores.var():.4f}')
        return None

    def log_model_scores(self):
        scores = pd.DataFrame(self.scores, columns=['Test R2', 'Test RMSE'])
        scores.to_csv(ROOT_DIR/ f'logs/ETa_{self.model_name}_k_scores.csv')
        return None

    def rescale_series(self, eta, index=None): 
        # Create fake DataFrame with fake features
        X = pd.DataFrame(columns=self.features)
        X['ETa'] = eta
        scaler = load(ROOT_DIR/'models'/'scaler.joblib')
        rescaled_eta = scaler.inverse_transform(X)[:, [-1]].ravel()
        if index is not None:
            # Create a DataFrame
            rescaled_eta = pd.DataFrame(rescaled_eta, columns=['ETa'], index=index)
        return rescaled_eta


@click.group()
@click.option('--log', is_flag=True, help='Log training to Neptune')
def make_model(*args, **kwargs):
    """
    Train a Machine Learning model and test it.
    Training and testing is implemented on k-folds of data.
    The mean score (R2) and score variance are returned.
    """
    return None


@click.command()
@click.option('--bootstrap', default=True)
@click.option('--ccp-alpha', type=click.FLOAT, default=0.0)
@click.option('--max-depth', type=click.INT, default=None)
@click.option('--max-samples', type=click.INT, default=None)
@click.option('-n', '--n-estimators', type=click.INT, default=100)
@click.option('--random-state', type=click.INT, default=6474)
def rf(**kwargs):
    model = RandomForestRegressor(**kwargs)
    model_name = 'rf'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    model = ModelTrainer(model, model_name)


@click.command()
@click.option('--activation', default='relu',
              type=click.Choice(
                  ['identity', 'logistic', 'tanh', 'relu'], 
                  case_sensitive=False))
@click.option('--solver', default='adam',
              type=click.Choice(
                  ['lbfgs', 'sgd', 'adam'], 
                  case_sensitive=False))
@click.option('--alpha', type=click.FLOAT, default=0.0001)
@click.option('--learning-rate', default='constant',
              type=click.Choice(
                  ['constant', 'invscaling', 'adaptive'], 
                  case_sensitive=False))
@click.option('--max-iter', type=click.INT, default=200)
@click.option('--shuffle', default=True)
@click.option('-hls', '--hidden-layer-sizes', 
              type=click.INT, default=[100,], multiple=True)
@click.option('--random-state', type=click.INT, default=12)
def mlp(**kwargs):
    model = MLPRegressor(**kwargs)
    model_name = 'mlp'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    model = ModelTrainer(model, model_name)
    return None


@click.command()
@click.option('--kernel', default='rbf',
              type=click.Choice(
                  ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 
                  case_sensitive=False))
@click.option('--degree', type=click.INT, default=3)
@click.option('--gamma', default='scale',
              type=click.Choice(
                  ['scale', 'auto',], 
                  case_sensitive=False))
@click.option('--tol', type=click.FLOAT, default=1e-3)
@click.option('--epsilon', type=click.FLOAT, default=0.1)
@click.option('--max-iter', type=click.INT, default=-1)
def svr(**kwargs):
    model = SVR(**kwargs)
    model_name = 'mlp'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    model = ModelTrainer(model, model_name)  
    return None


@click.command()
@click.option('--kernel', default=None,)
@click.option('--alpha', type=click.FLOAT, default=1e-10)
@click.option('--random-state', type=click.INT, default=6474)
def gpr(**kwargs):
    model = GaussianProcessRegressor(**kwargs)
    model_name = 'mlp'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    model = ModelTrainer(model, model_name)  
    return None


make_model.add_command(rf)
make_model.add_command(mlp)
make_model.add_command(svr)
make_model.add_command(gpr)


def main(model, model_name, features, visualize_errors=False):
    """ Train and save model """
    model = ModelTrainer(model, model_name, features, visualize_errors)
    return None

if __name__ == "__main__":
    rf()
