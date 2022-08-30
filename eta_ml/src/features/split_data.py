"""
Created by Federico Amato
2022 - 08 - 28

Split processed data in features and target.
Make a .pickle for every feature series and target
and a collective .pickle for all features.

"""
import pandas as pd
import os

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
SEED = 1998


def read_data(fname='processed.pickle'):
    fname = PROJ_ROOT + '/eta_ml/data/processed/' + fname
    # initialize data frame
    df = pd.DataFrame
    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError as e:
        print("File not found error: " + str(e))
    return df


def split_data(data):
    """Split data into a only-features set and a target set"""
    # ETa is the target series to predict
    trgt = data['ETa'].dropna().dropna()
    # Every column of data is a feature except from ETa
    fts = data.drop(labels='ETa', axis=1)
    return fts, trgt


def save_data(df1, fname1, df2, fname2):
    df1.to_pickle(PROJ_ROOT + '/eta_ml/features/' + fname1)
    df2.to_pickle(PROJ_ROOT + '/eta_ml/features/' + fname2)
    return None


def main():
    print(f"\n\n{'*'*10} START FEATURE EXTRACTION {'*'*10}\n\n")
    print(PROJ_ROOT, end="\n\n")

    # Read processed data
    df = read_data('processed.pickle')
    # Split features and target
    features, target = split_data(df)
    # Print data sets
    print(features.head(), end='\n\n')
    print(target.head(), end='\n\n')
    # Save features and target
    save_data(features, 'features.pickle', target, 'target.pickle')

    print(f"\n\n{'*'*10} END FEATURE EXTRACTION {'*'*10}\n\n")
    return None


if __name__ == "__main__":
    main()
