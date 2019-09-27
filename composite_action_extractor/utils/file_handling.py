import os
import queue
import pandas as pd
import numpy as np
from pathlib import Path, PosixPath


def get_recent_data_path(directory='data', project_root='composite_action_extractor', directory_hint=None,
                         mode='recent',
                         ignore_hidden=True, ignore_files=True):
    """
    Gets the path to the most recent data directory.

    Due to the possibility that there are different versions of a composite dataset, we might have different versions
    of the dataset or subsets of the dataset.

    Args:
        mode: Can be [recent, reverse, mid]
        directory: The data directory to evaluate.
        project_root: The directory that the absolute path finder will branch back down from.
        directory_hint: If not None, will try to evaluate directory names that contain it. Can be as specific or vague
                        as needed.
        ignore_hidden: Not supported yet
        ignore_files: Not supported yet

    Returns:

    """
    if not ignore_hidden:
        raise NotImplementedError('ignore_hidden is not supported currently.')
    if not ignore_files:
        raise NotImplementedError('ignore_files is not supported currently.')

    path_data_folder = get_absolute_path(directory, project_root, ignore_hidden, ignore_files)
    directory_names = os.listdir(path_data_folder)
    print(f'[get_recent_data_path] Found in {path_data_folder} : {directory_names}')

    if directory_hint:
        directory_names = [_ for _ in directory_names if str(_).__contains__(directory_hint)]
        print(f'[get_recent_data_path] Filtered in {path_data_folder} : {directory_names}')

    if mode == 'recent':
        found_name = sorted(directory_names).pop(0)
    elif mode == 'reverse':
        found_name = sorted(directory_names).pop(-1)
    else:
        found_name = sorted(directory_names).pop(len(directory_names) // 2)

    found_path = os.path.join(path_data_folder, found_name)
    print(f'[get_recent_data_path] Found {found_name}. Returning {found_path}')
    return found_path


def get_absolute_path(directory, project_root='composite_action_extractor', directory_file_hint=None,
                      ignore_hidden=True, ignore_files=True):
    """
    Gets the absolute path to a directory in the project structure using depth first search.

    Args:
        directory: The name of the folder to look in.
        project_root: The project root name. Generally should not be changed.
        ignore_hidden: For future use, for now, throws error because it cannot handle hidden files.
        ignore_files: For future use, for now
        , throws error because it is expecting directories.

    Returns:

    """
    if not ignore_hidden:
        raise NotImplementedError('ignore_hidden is not supported currently.')

    full_path = Path(__file__).parents[0]  # type: PosixPath

    # Move up the path address to the project root
    while full_path.name != project_root:
        full_path = full_path.parents[0]

    # Find the path to the directory
    searched_directory = queue.Queue()
    searched_directory.put_nowait(full_path)
    # Will do a depth first search
    while not searched_directory.empty():
        full_path_str = str(searched_directory.get_nowait())
        if not directory_file_hint and os.path.exists(os.path.join(full_path_str, directory)):
            return os.path.join(full_path_str, directory)

        if os.path.exists(full_path_str) and directory_file_hint and full_path_str.__contains__(directory_file_hint):
            return full_path_str

        for inner_dir in os.listdir(full_path_str):
            if str(inner_dir).__contains__('.'):
                if ignore_files:
                    continue  # Directory is either a file (not containing the hint), or hidden. Skip this
                if directory_file_hint and str(inner_dir).__contains__(directory_file_hint):
                    return os.path.join(full_path_str, inner_dir)
                else:
                    continue

            searched_directory.put_nowait(os.path.join(full_path_str, inner_dir))

    raise IOError(f'Path to {directory} not found.')


def get_data_df(csv_file_name):
    # Read in entropy csv
    base_path = get_absolute_path('data', directory_file_hint=csv_file_name.lower(), ignore_files=False)
    csv_path = os.path.join(base_path, 'state_action_data.csv')

    pre_data = pd.read_csv(csv_path, index_col=None)
    selected_episodes = pre_data['episode'].unique()  # Either do [some index] or [:]
    selected_episodes = [selected_episodes] if np.isscalar(selected_episodes) else selected_episodes
    data_df = pre_data[pre_data['episode'].isin(selected_episodes)]
    data_df_episodes = [data_df[data_df['episode'] == episode].copy() for episode in selected_episodes]
    return data_df_episodes, selected_episodes
