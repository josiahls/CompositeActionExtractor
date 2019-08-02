import numpy as np
import os
import pandas as pd

from segmentation.composite_action_extractor import CompositeActionExtractor
from utils.file_handling import get_absolute_path


def test_get_composite_actions():
    # Entropy Parameters
    csv_file_name = 'boxing'
    episodes = -1

    base_path = get_absolute_path('data', directory_file_hint=csv_file_name.lower(), ignore_files=False)
    csv_path = os.path.join(base_path, 'state_action_data.csv')

    pre_data = pd.read_csv(csv_path, index_col=None)
    selected_episodes = pre_data['episode'].unique()[episodes]  # Either do [some index] or [:]
    selected_episodes = [selected_episodes] if np.isscalar(selected_episodes) else selected_episodes
    data_df = pre_data[pre_data['episode'].isin(selected_episodes)]
    data_df_episodes = []
    for episode in selected_episodes:
        data_df_episodes.append(data_df[data_df['episode'] == episode].copy())

    main_df = data_df_episodes[episodes]

    CompositeActionExtractor.get_composite_actions(dataframe=main_df, analysis_method='kld')