"""
Created by Federico Amato
2022 - 08 - 29

Train Gaussian Process Regressor Model.
Save model and results.

Log run informations on neptune.ai

"""
import os
import pandas as pd

from gaussian_process import GaussianRegressionModel
# kernels
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel

# import sys
# sys.path.append('src')
# from data.preprocess import read_processed_data

import neptune.new as neptune

run = neptune.init(
    project='unipa-it-ml/ml-to-citrus-orchard-eta',
    tags="gaussian_process, fa",
    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzhiNjY2NC04ZDk1LTQ1MTktOWRkNy1mMDI1OTc5OTRjNzQifQ=='
)

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))


def make_kernel(**params):
    # Long term rising trend
    rbf_strength = params.get('rbf_stren')
    length_scale = params.get('rbf_len')
    long_term_trend_kernel = rbf_strength * RBF(length_scale=length_scale)
    # Seasonal variation
    length_scale1 = params.get('sine_rbf_len')
    length_scale2 = params.get('sine_len')
    seasonal_kernel = (
        2.0**2
        * RBF(length_scale=100.0)
        * ExpSineSquared(length_scale=length_scale, periodicity=1.0, periodicity_bounds="fixed")
    )
    # Small irregularities
    length_scale = params.get('irreg_len')
    irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=length_scale, alpha=1.0)
    # Noise
    noise_level = params.get('noise_level')
    noise_bounds = params.get('noise_bounds')
    noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
        noise_level=noise_level, noise_level_bounds=noise_bounds
    )
    return long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel


def kernel_params():
    kernel_params = {
        'rbf_stren': 10.0**2,
        'rbf_len': 360.0,
        'sine_rbf_len': 150.0,
        'sine_len': 30.0,
        'irreg_len': 30.0,
        'noise_level': 0.4**2,
        'noise_bounds': (1e-5, 1e5),
    }
    return kernel_params

def model_params(kernel):
    params = {
        'kernel': kernel,
        'alpha': 1e-9,
        'optimizer': 'fmin_l_bfgs_b',
        'n_restarts_optimizer': 2,
        'normalize_y':False,
    }
    return params


def main():
    print(f"\n\n{'//'*10} TRAINING MODEL {'//'*10}\n\n")
    print(PROJ_ROOT, end='\n\n')

    # Take processed data
    data = read_processed_data().dropna()
    # Gaussian Process Model
    kernel = make_kernel(**kernel_params())
    params = model_params(kernel)
    seed = 21979
    model = GaussianRegressionModel(data, seed, **params)
    model.train()

    test_score = model.test()
    print(f"Score: {test_score:.4f}")


    # Log to Neptune
    params['seed'] = seed
    run['parameters'] = params

    run['test/r2'] = test_score

    print(f"\n\n{'//'*10} END TRAINING {'//'*10}\n\n")
    return None

if __name__ == "__main__":
    main()
