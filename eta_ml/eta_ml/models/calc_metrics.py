# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:30:10 2023

@author: Federico Amato

Compute various metrics comparing postprocessed KC with:
    - Measured KC
    - Allen
    - Rallo
    - VI model
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from pathlib import Path
ROOT = Path(__file__).parent.parent.parent


def get_trapezoidal():
    df = pd.read_csv(ROOT / 'data/external' / 'trapezoidal_kc.csv',
                     sep=';', decimal=',',
                     index_col=0,
                     parse_dates=True, 
                     infer_datetime_format=True, dayfirst=True,
                     skiprows=[0])
    return df


def get_vi_model():
    df = pd.read_csv(ROOT / 'data/external/VIs_Kc_2018_2022.csv',
                     sep=';', decimal=',',
                     index_col=0,
                     parse_dates=True, 
                     infer_datetime_format=True, dayfirst=True,
                     )
    return df

def plot_models(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(ax=ax)
    plt.show()
    return None
    

def calc_errors(true, predictions, metric_name):
    df = pd.concat([true, predictions], axis=1, join='inner')
    rmse = mean_squared_error(df.iloc[:, 0], df.iloc[:, 1], squared=False)
    print(f'RMSE {metric_name}: {rmse:.4f}')
    return rmse


def main(filename):
    print(f'{"-"*5} COMPARE MODELS TO CALC METRICS {"-"*5}\n\n')
    
    predictions = pd.read_pickle(ROOT / 'data/predicted' / 'predicted.pickle')
    measures = predictions.loc[predictions['Source'] == 'Measured']['Kc']
    processed_kc = pd.read_pickle(ROOT / f'data/predicted/{filename}.pickle')['Kc']
    theoretical = get_trapezoidal()
    allen = theoretical.iloc[:, 0].to_frame()
    rallo = theoretical.iloc[:, 1].to_frame()
    vi = get_vi_model()
    
    calc_errors(measures, processed_kc, "Postprocessing on Measures")
    calc_errors(allen, processed_kc, "Postprocessing on Allen")
    calc_errors(rallo, processed_kc, "Postprocessing on Rallo")
    calc_errors(vi, processed_kc, "Postprocessing on Vi Kc")
    
    
    print(f'\n\n{"-"*21}')
    return None
    
    

if __name__ == "__main__":
    filename = 'kc_postprocessed'
    main(filename)
    

