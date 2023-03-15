# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:13:47 2022

@author: Federico Amato

Read CSV data and make Pickle File
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import logging

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent


def get_raw_data(fname):
    if str(fname).endswith('.csv'):
        df = pd.read_csv(fname, sep=';', decimal=',',
                         index_col=0, 
                         parse_dates=True, 
                         infer_datetime_format=True, 
                         dayfirst=True)
    elif str(fname).endswith('.xlsx'):
        df = pd.read_excel(fname, decimal=',',
                           index_col=0, parse_dates=True)
    df['SWC'] = df['SWC'].replace(0, np.NaN)
    df['Week'] = pd.Series(df.index).dt.isocalendar().week.values
    df['Month'] = df.index.month
    # NaN values in Precipitations and Irrigation are considered as null
    df['P'] = df['P'].fillna(0)
    df['I'] = df['I'].fillna(0)
    # Some column of the file may use dots as decimal
    # separators, so those columns are interpreted as 
    # object type and must be casted to floats.
    # See for example SWC, Tavg, RHavg
    return df.astype(np.float64)


def make_pickle(df, out):
    try:
        df.to_pickle(out)
    except Exception:
        print("Something went wrong writing Pickle file.\nTry again")
    

def main(input_file, output_file, visualize=True):
    logging.info(f'\n\n{"-"*5} MAKE DATA {"-"*5}\n\n')
    data = get_raw_data(input_file)
    logging.info(f'The file:\n'
          f'{input_file}\n'
          f'has the shape {data.shape} with columns:')
    for c in data.columns:
        logging.info(c)
    make_pickle(data, output_file)
    if visualize:
        data.plot(subplots=True, figsize=(10, 16))
        plt.savefig(ROOT_DIR / 'visualization/data' / 'raw_data.png')
    logging.info(f'\n\n{"-"*21}')
    return None


@click.command()
@click.option('-in', '--input-file',
              type=click.Path(exists=True),
              default=(ROOT_DIR/'data/raw'/'data.xlsx'),)
@click.option('-out', '--output-file', 
              type=click.Path(),
              default=(ROOT_DIR/'data/interim'/'data.pickle'),)
@click.option('-v', '--visualize', is_flag=True,)
def make_data(input_file, output_file, visualize):
    """
    Read raw CSV file and save the dataframe in a Pickle file.
    """
    main(input_file, output_file, visualize=visualize)
    return None


if __name__ == "__main__":
    input_file = ROOT_DIR / 'data/raw/db_villabate_deficit_6.csv'
    output_file = ROOT_DIR / 'data/interim/data.pickle'
    visualize = True
    main(input_file, output_file, visualize=visualize)