# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:47:04 2023

@author: Federico Amato
"""
from eta_ml.data.make_data import main as make_data
from eta_ml.data.preprocess import main as preprocess_data
from eta_ml.models.make_model import ModelTrainer
from eta_ml.prediction.predict import main as predict

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import itertools
import logging
import random
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent

# %% PARAMETERS

m0 = ['ETo', 'U2', 'RHmin', 'RHmax', 'Tmin', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY', 
      'I', 'P', 'EToC', 'IC', 'PC', 'LID', 'LPD', 'LWD']
m1 = ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY']
m2 = ['Rs', 'U2', 'RHmax', 'Tmin', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY']
m3 = ['Rs', 'U2', 'RHmax', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY']
m4 = ['Rs', 'U2', 'RHmax', 'Tmax', 'SWC', 'NDWI', 'DOY']
m5 = ['Rs', 'U2', 'Tmax', 'SWC', 'NDWI', 'DOY']
m6 = ['Rs', 'U2', 'Tmax', 'SWC', 'DOY']
m7 = ['Rs', 'Tmax', 'SWC', 'DOY']
m8 = ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin', 'Tmax']
m9 = ['ETo', 'SWC', 'NDVI', 'NDWI', 'DOY']
m10 = ['ETo', 'NDVI', 'NDWI', 'DOY']
m11 = ['Rs', 'SWC', 'NDVI', 'NDWI', 'DOY']
m12 = ['Rs', 'NDVI', 'NDWI', 'DOY']
m13 = ['Rs', 'U2', 'RHmin', 'RHmax', 'Tmin', 'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY', 'I', 'P']


FEATURES = {
    'model 1': m1,
    # 'model 2': m2,
    # 'model 3': m3,
    # 'model 7': m7,
    # 'model 9': m9,
    # 'model 10': m10,  
    }

MODELS = {
    'rf': RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            random_state=12,
            ccp_alpha=0.0,        
        ),
    # 'mlp': MLPRegressor(
    #         hidden_layer_sizes=(100, 100, 100),
    #         max_iter=1000,
    #         random_state=32652,  # 32652
    #     ),
    # 'knn': KNeighborsRegressor(
    #     n_neighbors=5,
    #     weights='distance',        
    #     ),
    }

MAKE_DATA_PARAMETERS = {
    'input_file': (ROOT_DIR / 'data/raw/db_villabate_deficit_9_2018_2021_irr.csv'),    
    'output_file': ROOT_DIR / 'data/interim/data.pickle',
    'visualize': True,
    }

PREPROCESS_PARAMETERS = {
    'input_file': ROOT_DIR / 'data/interim/data.pickle',
    'features': None,
    'scaler': 'MinMax',
    'folds': 4,
    'k_seed': 325,  # 24
    'output_file': ROOT_DIR / 'data/processed/processed.pickle',
    'visualize': False,
    }

PREDICTION_PARAMETERS = {
    # 'output': ROOT_DIR / 'data/predicted'/'predicted.pickle',
    'eta_output': ROOT_DIR / 'data/predicted' / 'eta_predicted.pickle',
    'features': None,
    'visualize': False,
    }


def setup_logging():
    # Config logging module
    logging.basicConfig(
        encoding='utf-8',
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(ROOT_DIR / "logs/et_predict.log"),
            logging.StreamHandler()
        ]
    )
    # Create a logging ID
    random_digits = str(random.randint(0, 99999)).zfill(5)
    random_character = random.choice(string.ascii_uppercase)
    log_id =  random_digits +random_character
    logging.info(f'__________________________________\n{str(datetime.datetime.now())}')
    logging.info(f"\nStarting run ID {log_id}")
    return log_id


def clean_processed_dir():
    # Remove train/test folds files from processed folder
    processed_dir = ROOT_DIR / 'data/processed'
    # Iterate over all files in the folder
    for file in processed_dir.iterdir():
        # Check if the file has a .pickle extension
        if file.suffix == ".pickle":
            # Delete the file using unlink()
            file.unlink()
    return None


def make_prediction(features_set, model_name, model_scores, kts):
    logging.info(f"\n{'*'*5} {model_name.upper()} {features_set.upper()} {'*'*5}\n")
    
    features = FEATURES[features_set]
    
    # Update parameters with model features
    PREPROCESS_PARAMETERS.update({'features': features})
    PREDICTION_PARAMETERS.update({'features': features})
    
    preprocess_data(**PREPROCESS_PARAMETERS)
    
    MODEL_PARAMETERS = {
            'model': MODELS[model_name],
            'model_name': model_name,
            'features': features,
            'visualize_error': False,
        }
    
    trainer = ModelTrainer(**MODEL_PARAMETERS)
    model_scores[model_name][features_set] = trainer.scores
    kts[model_name][features_set] = trainer.kt
    
    
    predict(model=f'{model_name}.joblib', **PREDICTION_PARAMETERS)
    
    return model_scores


def prettify_scores(scores):
    """
    Make a MultiIndex DataFrame from the nested dictionary of scores.

    """
    metrics_used = ['r2', 'rmse']
    # Create tuple keys for dictionary
    scores = {(outer_key, inner_key): values.ravel() for outer_key, inner_dict in scores.items() for inner_key, values in inner_dict.items()}
    # Create rows MultiIndex
    index = pd.MultiIndex.from_product(
        [[i+1 for i in range(PREPROCESS_PARAMETERS['folds'])], metrics_used],
        names=["fold", "metric"]
        )
    # Create DataFrame from dictionary
    scores = pd.DataFrame(scores)
    scores.index = index
    return scores

def reframe_kts(kts):
    # Take the temporal index
    idx = kts[list(MODELS.keys())[0]][list(FEATURES.keys())[0]].index
    # Create tuple keys for dictionary
    kts = {(model, feature_set): values.Kt.ravel() 
           for model, inner_dict in kts.items() 
           for feature_set, values in inner_dict.items()}
    kts = pd.DataFrame(kts, index=idx)
    return kts

def make_violins(kts, suptitle=None):
    models = kts.columns.get_level_values(0).unique()
    fig, axs = plt.subplots(len(models), figsize=(12, 6*len(models)))
    if suptitle is not None:
        fig.suptitle("Relative Error distribution across features sets", fontsize=18)
    else:
        fig.suptitle(suptitle, fontsize=18)
    for i, m in enumerate(models):
        if len(models) > 1:
            ax = axs[i]
        else:
            ax = axs
        ax.set_title(m.upper())
        data_to_plot = kts.loc[:, m]
        sns.violinplot(data_to_plot, ax=ax)
        ax.set_ylim(-1.5, 4.5)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, ls='--', c='k')
    plt.tight_layout()
    plt.show()
    return axs


# %% MAIN
def main():

    # Logging setup
    log_id = setup_logging()    
    # Clean directory with train/test data
    clean_processed_dir()
            
    make_data(**MAKE_DATA_PARAMETERS)
    # Models scores
    model_scores = {regressor: {} for regressor in MODELS}
    # Relative Errors for every model
    kts = {regressor: {} for regressor in MODELS}
    # Iterate over features sets and models    
    for model_name, features_set in itertools.product(MODELS, FEATURES):
        # Make cross-predictions and get the scores
        model_scores = make_prediction(features_set, model_name, model_scores, kts)

    # Reformat scores and errors in a DataFrame    
    model_scores = prettify_scores(model_scores)
    kts = reframe_kts(kts)
    # Violin Plot of Relative Errors
    make_violins(kts)
    
    # Save Scores
    model_scores.to_csv(ROOT_DIR / f'logs/eta_scores_{log_id}.csv', sep=';')
    # Save Relative Errors
    kts.to_csv(ROOT_DIR / f'data/predicted/kts_{log_id}.csv', sep=';')
    print("Process finished with log ID: ", log_id)
    
    return None
        

# %% Entry point
if __name__ == "__main__":
    main()
