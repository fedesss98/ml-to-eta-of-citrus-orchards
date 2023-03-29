# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 06:39:17 2023

@author: Federico Amato
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PLOTS_DIR = ROOT_DIR / 'visualization/errors/violins'

PLOTS_FONT_SIZE = 18


def ask_for_run_id():
    # Ask for an ID until it is find a Relative Errors file with that ID
    while True:
        run_id = input("Please provide an ID for the run: ")
        if not os.path.exists(ROOT_DIR / f'data/predicted/kts_{run_id}.csv'):
            print("File with this ID does not exists.")
        else: 
            return run_id
        

def read_kts(run_id):
    if run_id is None:
        run_id = ask_for_run_id()
    
    kts = pd.read_csv(ROOT_DIR / f'data/predicted/kts_{run_id}.csv', sep=';',
                      index_col=0, header=[0, 1],
                      parse_dates=True,
                      infer_datetime_format=True)
    return kts


def make_violins(kts, suptitle=None):
    data_to_plot = kts.copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=PLOTS_FONT_SIZE)
    sns.violinplot(data_to_plot, ax=ax)
    
    ax.set_ylim(-1.5, 3.1)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0, ls='--', c='k')
    ax.set_ylabel("Relative error [mm/day]", fontsize=PLOTS_FONT_SIZE*1.2)
    # Set size of fonts
    plt.xticks(size=PLOTS_FONT_SIZE)
    plt.yticks(size=PLOTS_FONT_SIZE)
    plt.tight_layout()
    plt.show()
    return ax


def save_violins(ax,  suffix='', sub_dir=None):
    saving_path = PLOTS_DIR / sub_dir if sub_dir is not None else PLOTS_DIR
    fig = ax.figure
    # PNG format
    fig.savefig(saving_path / f'violins_{suffix}_{run_id}.png', dpi=300)
    # PDF format
    fig.savefig(saving_path / f'violins_{suffix}_{run_id}.pdf')
    # EPS format
    fig.savefig(saving_path / f'violins_{suffix}_{run_id}.eps')
    return None


def main(run_id=None):
    
    kts = read_kts(run_id)
    
    # PLOT VIOLIN PLOTS
    # for every regressor used
    for regressor, models_data in kts.groupby(level=0, axis=1):
        models_data = models_data.droplevel(0, axis=1)
        models_data.columns = [regressor.upper() + model_name.strip('model ') 
                               for model_name in models_data.columns]
        ax = make_violins(models_data)
        save_violins(ax, suffix=regressor)
    
    # SELECT IRRIGATION SEASON.
    # Irrigation season starts at 15 May and ends 31 Sep
    irrigation_kt = kts.loc[(kts.index.dayofyear > 167) & (kts.index.month < 10)]
    
    # PLOT VIOLIN PLOTS
    # for every regressor used
    for regressor, models_data in irrigation_kt.groupby(level=0, axis=1):
        models_data = models_data.droplevel(0, axis=1)
        models_data.columns = [regressor.upper() + model_name.strip('model ') 
                               for model_name in models_data.columns]
        ax = make_violins(models_data)
        save_violins(ax, suffix=regressor, sub_dir='irrigation_season')


if __name__ == "__main__":
    # Use the log ID of the run you want to analyze
    run_id = '55323T'
    main(run_id)
