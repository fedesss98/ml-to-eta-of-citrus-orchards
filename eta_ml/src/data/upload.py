"""
Created by Federico Amato
2022 - 09 - 27

Copy the excel file with raw data in /data/raw/ project folder.
Copy the excel file from ORIGIN folder.
Creates .csv and .pickle file from excel and put them in
/data/interim/ project folder with name 'castelvetrano'

"""
import pandas as pd
import os
import shutil
from pathlib import Path
from datetime import datetime

# Change this path to point the raw file 'Database_Castelvetrano.xlsx'
HOME = str(Path.home())
ORIGIN = HOME + '/Scaricati/'
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))


def format_data(fname):
    df = pd.read_excel(fname,
                       header=0, skiprows=5, index_col=0,
                       usecols='A:C,E,G:P',
                       parse_dates=True,
                       decimal=','
                       )
    df.columns = [
        'NDVI', 'NDWI', 'ETa',
        'Tmin', 'Tmax', 'Tmean',
        'RHmin', 'RHmax', 'RHmean',
        'Rs', 'Ws', 'Rain', 'ET0']
    # Save pickle
    df.to_pickle(
        PROJ_ROOT + '/eta_ml/data/interim/' + 'castelvetrano.pickle')
    # Save csv
    df.to_csv(
        PROJ_ROOT + '/eta_ml/data/interim/' + 'castelvetrano.csv')
    return df


def main():
    print(f"\n\n{'_'*10}START PROJECT{'_'*10}\n\n")
    print(PROJ_ROOT)
    filename = 'Database_Castelvetrano.xlsx'

    # Copy excel file from ORIGIN
    origin_file = ORIGIN + filename
    project_file = PROJ_ROOT + '/eta_ml/data/raw/' + filename
    shutil.copy(origin_file, project_file)

    # Read excel and creates .csv and .pickle
    data = format_data(project_file)

    # Print some info about data
    print(data.head())
    data.plot(subplots=True, figsize=(12, 12))

    print(f"\n\n{'_'*10}END PROJECT{'_'*10}\n\n")
    return None


if __name__ == "__main__":
    main()
