# Machine Learning Models to predict daily actual evapotranspiration of citrus orchards
## A. Pagano; D. Croce; F. Amato; M. Ippolito; D. De Caro; A. Motisi; G. Provenzano; I. Tinnirello

Precise estimation of actual evapotranspiration is essential for various hydrological processes, including those related to agricultural water source management. 
Indeed, the increasing demands of agricultural production, coupled with increasingly frequent drought events in many parts of the world, necessitate a more careful assessment of irrigation needs. 

Artificial Intelligence-based models represent a promising alternative to the most common estimation techniques. 
In this context, the main challenges are choosing the best possible model and selecting the most representative features. 
The objective of this research was to evaluate two different machine learning algorithms, namely Multi-Layer Perceptron (MLP) and Random Forest (RF), to predict daily actual evapotranspiration in a Mediterranean citrus orchard using different feature combinations. With many features available coming from the various infield sensors, a thorough analysis was performed to measure feature importance (FI), scatter matrix observations, and Pearson's correlation coefficient calculation, which resulted in the selection of 12 promising feature combinations. 
Overall, 24 different models were developed and compared, evaluating the performance of the prediction algorithm (both for RF or MLP) and the importance of the different input variables adopted. 

Results show that the accuracy of the proposed machine learning models remains acceptable even when the number of input features is reduced from 10 to 4. 
Among the different ML algorithms developed, the best performance was achieved by the Random Forest method when using seven input features. 
In this case, the values of the root mean squared error (RMSE) and coefficient of determination (R2) associated with cross-validation were 0.39 mm/d and 0.84, respectively. 

Finally, the results obtained show that the joint use of agro-meteorological and remote sensing data improves the performance of evapotranspiration forecasts compared with models using only weather variables. 

## Installation
To use this project in your machine you can download all files and save them to your local project folder, 
or you can clone this repository.
To clone the repository you should open the terminal inside your local project folder,
and input:

```console
$ git clone https://github.com/fedesss98/ml-to-eta-of-citrus-orchards.git
```

Now your folder is populated by all files coming from this repository.

### Folders Structure
Be sure to have this structure in your project:

<pre>
ROOT
|— eta_ml\
|   |- data\
|   |   |- external\
|   |   |- raw\
|   |   |- interim\
|   |   |- processed\
|   |   |- predicted\
|   |\
|   |- models\
|   |\
|   |- prediction\
|   |\
|   |- visualization\
|   |\
|   |- eta_ml\
|   |   |- data\
|   |   |- models\
|   |   |- prediction\
|   |   |- visualization\
|   |
|   
</pre>

Use the file `environment.yml` to install dependencies using `pip` or `conda`.

Run the script `eta_predict.py` inside `eta_ml` folder to generate predictions and trained Machine Learning models.
In "[Running the Code - Overview](#running-the-code---overview)" you will find this project workflow to train ML models and generate predictions. 
Details on code's inputs will be given in the Section "[Parameters](#parameters)".

Note in `eta_predict.py` both positional arguments and named arguments are gathered in dictionaries with uppercase names, and that `main` functions of every `eta_ml` submodule are imported with the name of the file to explicit their function.

## Running the Code - Overview
### Setup Data
This projects needs data to run.

The module `eta_ml/data` takes care of formatting and preprocessing data.

Raw data can be in the form of a csv or pickle file.
This file should contain various **features** and one **target** as 
columns, and days as row indices.
The snippet `make_data.py` in this module will read the file and format it, 
and eventually plot the time series.

Data must then be preprocessed to be feeded to Machine Learning models.
The code `preprocess.py` acts as follows:
- Select features,
- Impute missing values with a [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html),
- Split data in k-folds of train and test sets,
- Scale data with a [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html),

To never let the training set see the test set, the k-folds splitting must come before the scaling.
Each scaled fold of train-test data is then saved in pickle format. 

### Train the Model
Each ML model is an instance of the ModelTrainer class defined in `make_model.py` script in the `eta_ml/models` module.
ModelTrainer objects require a Scikit-learn Regressor model[^1], a custom name for that model and the list of features to use in training.
On creation, the model will be trained for each train-set saved fold and scores will be computed.

Considered scores are $R^2$ coefficient and Mean Squared Error, both given by Scikit-learn builtin functions form `sklearn.metrics`[^2].
Both compare real measured data and predictions, rescaled using the inverse transformation of the same scaler used in preprocessing.

The best scoring fold is chosen and the model trained with those data is saved with its given name as a joblib file.

Pointwise relative errors will also be computed fold by fold and gathered in the kt attribute of the ModelTraier --a pandas DataFrame--.

Using different Scikit-learn Regressors one can make different experiments and keep tracks of scores, which are saved fold by fold.

Having a trained model saved, it can be used to make predictions from input data. 

### Use the Model
The `predict.py` code in `eta_ml/prediction` module handles the prediction of missing data inside the time series of the target.

Taking away measured target data days --used to train and test the model--, remaining days features are given to one trained model to predict the target.

The time series of the target is thus completed by those predicted data, rescaled and saved.

## Parameters
Here follows the list of arguments for each script in `eta_ml` module.
The `main` function of each script will accept "Arguments" and "Options".
In what follows "Arguments" are positional arguments to give to the `main` function of each script,
while "Options" are optional keyword arguments.  
### make_data.py
<pre>
Arguments
_________
input_file [string or pathlib.Path]: position of the raw data file,
output_file [string or pathlib.Path]: position of the output formatted file.

Options
_______
visualize [boolean] (default = True): indicates wether to plot time series of the data or not. 
</pre>

It is advised to have raw data inside the `data/raw` folder inside your project root and save formatted data in `data/interim` folder.
### preprocess.py
<pre>
Arguments
_________
input_file [string or pathlib.Path]: position of the raw data file,
scaler [string]: choose between 'Standard' to use a StandardScaler or 'MinMax' for the MinMaxScaler,
folds [int]: number of folds to split the data,
k_seed [int]: seed to initialize the KFold splitter,
output_file [string or pathlib.Path]: position of the unsplitted processed data file.

Options
_______
features [list] (default = ['DOY', 'Rs', 'Tavg', 'ETo']): list of feature names to use,
visualize [boolean] (default = True): indicates wether to plot time series of the data or not. 
</pre>

It is advised to take input data from `data/interim` and save processed data in `data/processed` folder.
Train-test folds will automatically be saved inside `data/preprocessed` folder with name `train_fold_{k}.pickle` and `test_fold_{k}.pickle`, where `{k}` is current fold number.
If the plot of preprocessed time series is visualized, it is automatically saved in `visualization/data` folder.

### make_model.py
<pre>
Arguments
_________
model [sklearn regressor]: Scikit-learn regressor that implements "fit" and "predict" methods,
model_name [string]: human-readable name to distinguish the model,
features [list]: list of features to use in training and predicting.

Options
_______
visualize_errors [boolean] (default = False): indicates wether to plot time series of the relative errors between measures and predictions.
</pre>

This script will create an instance of the class ModelTrainer, that holds different attributes:
- the r2 score and mean squared error for every fold inside the `scores` attribute,
- the DataFrame of relative errors time series for every fold inside the `kt` attribute.

### predict.py
<pre>
Arguments
_________
model [string]: Name of the saved trained model. If none is given, it is asked to input it in the console. 
eta_output [string or pathlib.Path]: position of the rescaled time series of target measures and predictions file.

Options
_______
features [list] (default = all features): list of features to use for predictions,
visualize [boolean] (default = True): indicates wether to plot target predictions and measures and confront them in a linear measures vs predictions plot.
</pre>

---

[^1]: We used as models the [Multy-layer Perceptron Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) and [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
[^2]: Metrics documentation can be found following these links: [R2 coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html),
[mean squared error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html).



