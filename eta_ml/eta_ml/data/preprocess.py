# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:03:31 2022

@author: Federico Amato

Preprocess ETa data for predictions.

- Select features
- Impute missing values with KNNImputer
- Split data with KFolds
- Scale data with StandardScaler or MinMaxScaler

Train set must never see test set, even during the scaling.
Thus the k-folds slpit must come before the scaling.

FREE PARAMS:
    - NN: number of neighbors for KNNInputer
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import joblib  # To save scaler
import json  # To save log
import pandas as pd
import logging

from sklearn.impute import KNNImputer
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from .make_data import  make_pickle

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent

NN = 5  # Number of neighboring samples to use for imputation


def get_data(fname):
    df = pd.read_pickle(fname)
    return df


def get_features(df, features):
    if len(features) <= 0:
        features = [
            'Rs',
            'Tavg',
            'DOY',
            'ETo'
            ]
    for f in features:
        if f not in df.columns:
            print(f"Feature {f} not present in raw data features!")
            features.remove(f)
    return df.loc[:, features]


def get_target(df):
    target = ['ETa']
    return df.loc[:, target]


def make_dataframe(data, features):
    """Selects only relevant features (Model #1 in previous experiments)"""
    features = get_features(data, features)
    target = get_target(data)
    # Concatenate features and target data
    df = pd.concat([features, target], axis=1)
    return df


def make_scaler(scaler):
    if isinstance(scaler, str):
        if scaler.lower() == 'standard':
            scaler = StandardScaler()
        elif scaler.lower() == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        try:
            scaler.set_params()
        except Exception as e:
            raise Exception(f'Error with the scaler:\n{str(e)}')
    return scaler


def impute_features(df, features):
    features = get_features(df, features)
    # Impute missing features
    # imputer = KNNImputer(n_neighbors=NN, weights='distance')
    imputer = IterativeImputer(random_state=0)
    imputed_values = imputer.fit_transform(features)
    # Recreate imputed DataFrame inserting target column
    # Take numpy array of ETa values
    eta_values = df['ETa'].values.reshape(-1, 1)
    # and merge it with imputed feature values
    imputed_values = np.append(imputed_values, eta_values, axis=1)
    # Make DataFrame
    imputed_data = pd.DataFrame(imputed_values, 
                                columns=df.columns, index=df.index)
    return imputed_data


def split_folds(df, k, k_seed=2):
    folds = KFold(k, shuffle=True, random_state=k_seed)
    df = df.dropna()
    for k, [train_index, test_index] in enumerate(folds.split(df)):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        make_pickle(train, ROOT_DIR/'data/processed'/f'train_fold_{k}.pickle')
        make_pickle(test, ROOT_DIR/'data/processed'/f'test_fold_{k}.pickle')


def scale_data(df, scaler):
    """
    Scale data using selected scaler:
    - Standard Scaler (zero mean and unit variance) [DEFAULT]
    - MinMax Scaler (values between minus one and plus one)
    """
    scaled_data = df.copy()
    try:
        # Fit and save scaler
        scaler.fit(df)
        joblib.dump(scaler, ROOT_DIR/'models'/'scaler.joblib')
        scaled_values = scaler.transform(df)
        scaled_data = pd.DataFrame(scaled_values, 
                                   columns=df.columns, index=df.index)
    except Exception as e:
        print(f'Error with the scaler: {str(e)}')
        scaled_data = df.copy()
    return scaled_data


# def write_log(input_file, scaler, output_file, visualize, df, train, test):
#     json_log = {
#         "input_file": input_file,
#         "output_file": output_file,
#         "scaler": scaler,
#         "database_shape": df.shape(),
#         "train_set_length": len(train),
#         "test_set_length": len(test),
#     }
#     json.dump(json_log, ROOT_DIR/'docs'/'run.json')


def main(input_file, scaler, folds, k_seed, output_file, features=[], visualize=True):
    logging.info(f'\n\n{"-"*5} PREPROCESSING {"-"*5}\n\n')
    logging.info(f"Preprocessing file:\n{input_file}")
    data = get_data(input_file)
    df = make_dataframe(data, features)
    scaler = make_scaler(scaler)
    if visualize:
        df.plot(subplots=True, figsize=(10, 16))
        plt.show()        
    # IMPUTE
    df = impute_features(df, features)
    # SAVE AND VISUALIZE TOTAL DATAFRAME
    make_pickle(df, ROOT_DIR/'data/interim'/'imputed.pickle') 
    if visualize:
        df.plot(subplots=True, figsize=(10, 16))
        plt.savefig(ROOT_DIR/
                    'visualization/data'/
                    f'processed_{NN}_{scaler}.png')
        plt.show()
    # SAVE DATA TO PREDICT
    predict = df.loc[~df.index.isin(df.dropna().index), features]
    predict = scale_data(predict, scaler)
    make_pickle(predict, ROOT_DIR/'data/processed'/'predict.pickle')
    # SPLIT DATA TO TRAIN - TEST
    split_folds(df, folds, k_seed)
    # Iterate over folds
    for k in range(folds):
        train_file = ROOT_DIR/'data/processed'/f'train_fold_{k}.pickle'
        test_file = ROOT_DIR/'data/processed'/f'test_fold_{k}.pickle'
        train = pd.read_pickle(train_file)
        test = pd.read_pickle(test_file)
        # SCALE FOLD
        train = scale_data(train, scaler)
        test = scale_data(test, scaler)
        # SAVE FOLD
        make_pickle(train, train_file)
        make_pickle(test, test_file)
    # write_log(input_file, scaler, output_file, visualize, df, train, test)
    df = scale_data(df, scaler)
    make_pickle(df, output_file)
    logging.info(f'\n\n{"/"*30}')


@click.command()
@click.option('-in', '--input-file',
              type=click.Path(),
              default=(ROOT_DIR /'data/interim'/'db_villabate.pickle'),)
@click.option('-s', '--scaler', default='MinMax', 
              type=click.Choice(['Standard', 'MinMax'], case_sensitive=False))
@click.option('-k', type=click.INT, default=5, help="Number of folds")
@click.option('--k-seed', type=click.INT, default=2, help="Number of folds")
@click.option('-out', '--output-file', 
              type=click.Path(),
              default=(ROOT_DIR/'data/processed'/'processed.pickle'),)
@click.option('-v', '--visualize', is_flag=True,)
def preprocess_data(input_file, scaler, k, k_seed, output_file, visualize):
    """
    Preprocess ETa data for predictions.

    - Select features
    - Impute missing values with KNNImputer
    - Split data with KFolds
    - Scale data with StandardScaler or MinMaxScaler

    Train set must never see test set, even during the scaling.
    Therefore, K-folds slpit must come before the scaling.

    """
    main(input_file, scaler, k, k_seed, output_file, visualize=visualize)
    

if __name__ == "__main__":
    preprocess_data()
