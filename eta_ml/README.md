# ETa predictions via ML methods

## Models
- Random Forest ([ScikitLearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))
- MultiLayer Perceptron ([ScikitLearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html))
- !!! ToDo: Gaussian Process Regression ([ScikitLearn](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html))

## Structure
- **data** <--- Data used;
- --data/raw <--- Raw excel file with measures;
- --data/interim <--- Intermediate processed used (.csv and .pickle files, easy to read);
- --data/processed <--- Final processed data used (scaled with [Standard Scaler](https://scikit-learn.org/stable/modules/preprocessing.html));
- **models** <--- Trained models saved for later use;
- **notebooks** <--- Jupyter notebooks to quickly work on data;
- **logs** <--- Logs of every run and results;
- **eta_ml** <--- Actual Python files to populate other folders;
- **tests** <--- Raw Python files to test code;
- **visualization** <--- Figures to visualize data
