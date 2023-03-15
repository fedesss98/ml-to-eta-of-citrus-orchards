# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:06:00 2022

@author: Federico Amato

Utilities functions and classes
- Custom data type for click option: mlp hidden_layer_sizes
"""

def rf_arg_parser(arg):
    name, value = arg
    if name == 'n_estimators':
        return int(value)
    elif name == 'bootstrap': 
        return bool(value)
    elif name ==  'ccp_alpha':
        return float(value)
    elif name == 'max_depth':
        return int(value)
    elif name == 'min_samples_split':
        return float(value)
    elif name == 'random_state':
        return int(value)
    elif name == 'max_samples':
        return float(value)
    else:
        raise ValueError(f'Argument "{name}" not known')


def mlp_arg_parser(arg):
    name, value = arg
    if name == 'hidden_layer_sizes':
        return tuple(value)
    elif name == 'activation':
        if value in ['identity', 'logistic', 'tanh', 'relu']:
            return str(value)
        else:
            raise ValueError(f'Activation function "{value}" not known')
    elif name == 'solver':
        if value in ['lbfgs', 'sgd', 'adam']:
            return str(value)
        else:
            raise ValueError(f'Solver "{value}" not known')
    elif name == 'alpha':
        return float(value)
    elif name == 'learning_rate':
        if value in ['constant', 'invscaling', 'adaptive']:
            return str(value)
        else:
            raise ValueError(f'Learning rate "{value}" not known')
    elif name == 'max_iter':
        return int(value)
    elif name == 'shuffle':
        return bool(value)
    elif name == 'random_state':
        return int(value)
    else:
        raise ValueError(f'Argument "{name}" not known')
