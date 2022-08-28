# ETa predictions via ML methods

## Models
- Gaussian Process Regression ([ScikitLearn](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html))

## Structure
- **data** <--- Data used;
- --data/raw <--- Raw excel file with measures;
- --data/interim <--- Intermediate processed used (.csv and .pickle files, easy to read);
- --data/processed <--- Final processed data used (scaled with [Standard Scaler](https://scikit-learn.org/stable/modules/preprocessing.html));
- **models** <--- Trained models saved for later use;
- **notebooks** <--- Jupyter notebooks to quickly work on data;
- **reports** <--- Figures and tables with results to use for pubblication;
- **src** <--- Actual Python files to populate other folders;
- **tests** <--- Raw Python files to test code;
- **visualization** <--- Figures to visualize data
