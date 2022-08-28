import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))


def read_data(fname):
    """Read data from a csv or pickle file"""
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
    try:
        scaler = StandardScaler().fit(data)
        # Print data means and variances
        print_scaler_params(scaler)
        # Scale data
        data = scale_and_frame(data, scaler)
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


def save_data(dataframe, fname='processed.pickle'):
    fname = PROJ_ROOT + '/eta_ml/data/processed/' + fname
    dataframe.to_pickle(fname)


def main():
    print(f"\n\n{'_'*10}START CODE{'_'*10}\n\n")
    print(PROJ_ROOT, end='\n\n')
    input_file = PROJ_ROOT + '/eta_ml/data/interim/' + 'castelvetrano.pickle'

    # Make data frame
    df = read_data(input_file)
    # Preprocess data
    print("Preprocessing: Standard Scaler")
    df = preprocess_data(df)
    print(df.head())
    # Plot scaled data
    df.plot(subplots=True, figsize=(12, 12))
    # Save scaled dataframe
    output_file = 'processed.pickle'
    save_data(df, output_file)

    print(f"\n\n{'_'*10}END CODE{'_'*10}\n\n")


if __name__ == "__main__":
    main()
