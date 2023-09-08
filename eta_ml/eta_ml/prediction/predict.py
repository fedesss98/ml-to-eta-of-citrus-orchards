# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 18:43:00 2022

@author: Federico Amato

Predict ETa with saved model.
From measured and predicted ETa computes KC dividing by measured ET0.

"""
import click
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # To load model and scaler
import seaborn as sns

import logging

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent


def ask_for_model():
    """ If no model name is given, looks for saved models and ask to choose one """
    saved_models = list(ROOT_DIR.glob('models/*'))
    print(f"You have {len(saved_models)} saved:")
    for m in saved_models:
        print(m)
    model_name = input("Which model do you want to use? ")
    return str(model_name)


def load_model(model_name):
    if model_name is None:        
        model_name = ask_for_model()
    return joblib.load(ROOT_DIR/'models'/f'{model_name}')


def fill_eta(eta):
    measured = pd.read_pickle(ROOT_DIR/'data/processed'/'processed.pickle')
    eta = pd.concat([eta, measured['ETa']], axis=1)
    eta.rename(columns={'ETa':'ETa Measured'}, inplace=True)
    idx_predict = eta['ETa Predicted'].dropna().index
    # Combine series
    total_eta = pd.DataFrame()
    total_eta['ETa'] = eta.iloc[:, 0].fillna(eta.iloc[:, 1])
    total_eta['Source'] = ['Predicted' if idx in idx_predict else 'Measured'
                     for idx in eta.index]
    return total_eta


def rescale_series(eta): 
    # Reset original DataFrame with feature measures and predicted target
    df = pd.read_pickle(ROOT_DIR/'data/processed'/'processed.pickle')
    df['ETa'] = eta['ETa']
    scaler = joblib.load(ROOT_DIR/'models'/'scaler.joblib')
    rescaled_df = scaler.inverse_transform(df)
    df = pd.DataFrame(rescaled_df, columns=df.columns, index=df.index)
    eta['ETa'] = df['ETa'].to_frame()
    eto = df['ETo'].to_frame()
    return eta, eto


def plot_prediction(df, series_name, title=None):
    g = sns.relplot(
        data=df,
        x='Day',
        y=series_name,
        hue='Source',
        height=6,
        aspect=1.6,
        )
    if title is not None:
        g.fig.suptitle(title)
    plt.show()
    return None
    
    
def plot_linear(model, measures, features):
    X = measures.loc[:, features]
    y_measured = measures['ETa'].values
    y_predicted = model.predict(X)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.scatter(y_measured, y_predicted, c='k')
    ax.plot([-1, 0, 1], [-1, 0, 1], 'r--')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True)
    plt.show()
    return None


def main(model, eta_output, features=None, visualize=True):
    logging.info(f"\n{'-'*7} PREDICT ETa {'-'*7}\n\n")
    # Features to predict ETa
    X = pd.read_pickle(
        ROOT_DIR/'data/processed'/'predict.pickle')
    if features is not None:
        X = X.loc[:, features]
    measures = pd.read_pickle(
        ROOT_DIR/'data/processed'/'processed.pickle').dropna()
    # Predict ETa
    try:
        model = load_model(model)
        logging.info(f'Predicting from features:\n'
                     f'{X.columns.tolist()}')
        eta_predicted = model.predict(X)
        # Make a DataFrame of predictions
        eta = pd.DataFrame(
            eta_predicted, columns=['ETa Predicted'], index=X.index)
    except FileNotFoundError:
        logging.error("Error finding the model. Remember to include file extension.")
    # Make ETa DataFrame with measures and predictions
    eta = fill_eta(eta)
    if visualize:
        plot_prediction(eta, 'ETa', 'Measured and Predicted ETa (scaled)')
        plot_linear(model, measures, features)
    # Save ETa
    pd.to_pickle(eta, eta_output)
    logging.info(f'Predictions saved in:\n{eta_output}')
    logging.info(f'\n\n{"/"*30}\n\n')
    return None


@click.command()
@click.option('-m', '--model', prompt="Which model do you want to use?", 
                help="Name of saved model")
@click.option('-out', '--output-file', 
              type=click.Path(),
              default=(ROOT_DIR/'data/predicted'/'predicted.pickle'),)
@click.option('-v', '--visualize', is_flag=True)
def predict(model, output_file, visualize):
    """
    Predict ETa with given model.
    From measured and predicted ETa computes KC dividing by measured ET0.
    """
    main(model, output_file, visualize)


if __name__ == "__main__":
    predict()



