"""
Runs a list of parameters of the extractor and outputs a resulting csv.
"""
import os

from segmentation.composite_action_extractor import CompositeActionExtractor
from utils.file_handling import get_data_df
from timeit import default_timer as timer
import pandas as pd
import numpy as np

environments = [
    'Cartpole',
    'Mountaincar',
    'Pendulum',
    'Acrobot',
    # 'Boxing',
    # 'Breakout',
    # 'Pong',
    # 'Skiing',
    # 'Tennis',
]

methods = [
    # 'shannon',
    # 'approximate',
    'sample',
    # 'permutation',
    # 'jensen_shannon',
    # 'total_variance',
    # 'kl_divergence',
    # 'wasserstein',
]

existing_df = pd.read_csv('../data/numerical_results.csv') if os.path.exists('../data/numerical_results.csv') else None

for env in environments:
    df_episodes, found_episodes = get_data_df(env)

    for method in methods:

        row = {
            'Avg N Composite Actions': [],
            'Avg Composite Action Length': [],
            'N Episodes': len(df_episodes),
            'Avg Per Episode Iter': [],
            'Avg Time': [],
            'Method': method,
            'Env': env
        }

        for i, working_df in enumerate(df_episodes):

            start = timer()
            actions, info = CompositeActionExtractor.get_composite_actions(dataframe=working_df, analysis_method=method,
                                                                           window_size=5, bins=64)
            end = timer()

            actions = list(actions.values())[0]

            print(f'Testing Episode {i} using {method} in {env}')

            row['Avg Time'] += [end - start]
            row['Avg N Composite Actions'] += [len(actions)]
            if len(actions) != 1:
                row['Avg Composite Action Length'] += [len(action[~np.isnan(action)]) for action in actions]
            else:
                row['Avg Composite Action Length'] += [0]
            row['Avg Per Episode Iter'] += [len(working_df)]

        row['Avg Time'] = [np.around(np.average(row['Avg Time']), decimals=2)]
        row['Avg N Composite Actions'] = [np.around(np.average(row['Avg N Composite Actions']), decimals=2)]
        row['Avg Composite Action Length'] = [np.around(np.average(row['Avg Composite Action Length']), decimals=2)]
        row['Avg Per Episode Iter'] = [np.around(np.average(row['Avg Per Episode Iter']), decimals=2)]
        row_df = pd.DataFrame(data=row)
        existing_df = pd.concat([existing_df, row_df.copy()]) if existing_df is not None else row_df

existing_df.to_csv('../data/numerical_results.csv')
