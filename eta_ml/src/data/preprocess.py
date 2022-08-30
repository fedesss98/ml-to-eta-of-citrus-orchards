"""
Created by Federico Amato
2022 - 08 - 27

Preprocess data.

Take raw data in pickle or csv format (found in /data/interim)
and preprocess them with:
- Standard scaling to regularize data subtracting mean and dividing
  by variance
- Drop dates with missing features
Save final data in /data/processed/processed.pickle

"""
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))


def read_processed_data(fname='processed.pickle'):
    fname = PROJ_ROOT + '/eta_ml/data/processed/' + fname
    df = pd.read_pickle(fname)
    return df


def get_features(data):
    features = data.drop(labels='ETa', axis=1)
    return features


def get_target(data):
    target = data['ETa']
    return target


def read_data(fname):
    """Read raw data from a csv or pickle file"""
    # Get data file extension
    _, ext = os.path.splitext(fname)
    # Initialize data frame
    df = pd.DataFrame()
    try:
        if ext == '.csv':
            df = read_csv(fname)
        elif ext == '.pickle':
            df = read_pickle(fname)
    except FileNotFoundError as e:
        df = None
        print("Error reading file: " + str(e))
    return df


def read_csv(fname):
    df = pd.read_csv(fname, sep=';', decimal=',',
                     index_col=0,
                     parse_dates=True, infer_datetime_format=True,
                     )
    return df


def read_pickle(fname):
    df = pd.read_pickle(fname)
    return df


def preprocess_data(data):
    """Scale data dataframe with Scikit Standard Scaler"""
    print(data)
    try:
        scaler = StandardScaler().fit(data)
        # Print data means and variances
        print_scaler_params(scaler)
        # Scale data
        data = scale_and_frame(data, scaler)
        save_data(data, processed=False, fname='scaled.pickle')
        # Drop na
        data = drop_na(data)
    except Exception as e:
        print("Error in scaling: " + str(e))
    return data


def print_scaler_params(scaler):
    """Print means and variances used for the Standard Scaler"""
    params = pd.DataFrame([scaler.mean_, scaler.var_]).T
    params.columns = ['Mean', 'Variance']
    print(params, end='\n\n')


def scale_and_frame(df, scaler):
    columns = df.columns
    index = df.index
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns=columns)
    df.index = index
    return df


def drop_na(df):
    """Drop only features missing values dates"""
    features = get_features(df).columns
    df = df.dropna(subset=features)
    return df


def save_data(dataframe, processed=True, fname='processed.pickle'):
    folder = 'processed/' if processed else 'interim/'
    fname = PROJ_ROOT + '/eta_ml/data/' + folder + fname
    dataframe.to_pickle(fname)


def main():
    print(f"\n\n{'='*10} START PREPROCESSING {'='*10}\n\n")
    print(PROJ_ROOT, end='\n\n')
    input_file = PROJ_ROOT + '/eta_ml/data/interim/' + 'castelvetrano.pickle'

    # Make data frame
    df = read_data(input_file)
    # save length for output
    features_len = len(df)
    target_len = len(df['ETa'].dropna())

    # Preprocess data
    print("Preprocessing: scaling and dropping...")
    df = preprocess_data(df)
    # save length for output
    features_processed_len = len(df)
    target_processed_len = len(df['ETa'].dropna())

    # Plot scaled data
    df.plot(subplots=True, figsize=(12, 12))
    # Save scaled dataframe
    save_data(df)

    print(f"Number of points before and after preprocessing:\n"
          f"  Before:\n"
          f"  - Features: {features_len}\n"
          f"  - Target: {target_len}")
    print(f"  After:\n"
          f"  - Features: {features_processed_len}\n"
          f"  - Target: {target_processed_len}")

    print(f"\n\n{'='*10} END PREPROCESSING {'='*10}\n\n")


if __name__ == "__main__":
    main()
