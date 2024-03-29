from collections import Counter
from functools import partial

import pandas as pd
from typing import Tuple
import logging
import time

from composite_action_extractor.utils.divergence_distance_functions import *
from composite_action_extractor.utils.entropy_functions import *
from composite_action_extractor.utils.misc import bcolors, list_of_str_to_num, bin_numpy


class CompositeActionExtractor:
    registered_method_names = {'shannon': 's', 'approximate': 'ae', 'sample': 'se', 'permutation': 'p',
                               'jensen_shannon': 'js', 'total_variance': 'tv', 'kl_divergence': 'kld',
                               'wasserstein': 'w'}
    registered_method = {'shannon': entropy_shannon, 'approximate': entropy_approximate,
                         'sample': entropy_sample, 'permutation': entropy_permutation,
                         'jensen_shannon': jensen_shannon_divergence, 'total_variance': total_variation_distance,
                         'kl_divergence': kl_divergence, 'wasserstein': earth_mover_wasserstein_distance}

    @staticmethod
    def get_composite_actions(actions: np.array = None, state: np.array = None, dataframe: pd.DataFrame = None,
                              analysis_method='shannon', keep_percent=0.5, window_size=5, bins=64,
                              action_df_col_prefix='action_value', state_df_col_prefix='state_values', k=-1,
                              is_single_action=False) -> Tuple[dict, dict]:
        """
        Calculates segments for all actions using action, and state analysis information.

        Will also return latent information.

        Args:
            actions:
            state:
            dataframe:
            analysis_method: Choices are:
            {'shannon': 's', 'approximate': 'ae', 'sample': 'se', 'permutation':'p', \
                  'jensen_shannon': 'js', 'total_variance': 'tv', 'kl_divergence': 'kld', \
                 'wasserstein': 'w'}
            keep_percent:
            window_size:
            bins:
            action_df_col_prefix:
            state_df_col_prefix:

        Returns: (composite_actions, info)

        """
        logging.debug('Loading Data')
        if actions is None or state is None:
            if dataframe is None:
                message = bcolors.FAIL
                message += f'Parameters actions (is None?):{actions is None} or state (is None?):{state is None}.\n'
                message += f'If one of these is None, then dataframe cannot be None (was detected as None.)\n'
                message += bcolors.ENDC
                raise ValueError(message)
            else:
                start = time.time()
                actions = dataframe[[_ for _ in dataframe.columns if _.__contains__(action_df_col_prefix)]].values
                string_state = dataframe[[_ for _ in dataframe.columns if _.__contains__(state_df_col_prefix)]].values
                state = np.array([list_of_str_to_num(_)[0] for _ in string_state])
                if type(actions[0][0]) is str:
                    actions = np.array([list_of_str_to_num(_)[0] for _ in actions])
                end = time.time()
                logging.debug(f'Stage 1 [State/Action into Usable Inputs]: Elapsed time {end - start}')
        else:
            if dataframe is not None:
                message = bcolors.WARNING
                message += 'Dataframe is not None, but you are passing actions and state parameters. '
                message += f'We are going to ignore the dataframe.{bcolors.ENDC}'

        # Generates binned and windowed versions of the matrices.
        start = time.time()
        state_bw = CompositeActionExtractor._bin_p_window(state, window_size, bins, False)
        actions_bw = CompositeActionExtractor._bin_p_window(actions, window_size, bins, True and not is_single_action)
        end = time.time()
        logging.debug(f'Stage 2 [Bin / Prob / Window]: Elapsed time {end - start}')

        # Produce Analysis of both state and actions.
        start = time.time()
        state_analyzed = CompositeActionExtractor._analyze(state_bw, method=analysis_method, collapse_by_average=True)
        action_analyzed = CompositeActionExtractor._analyze(actions_bw, method=analysis_method,
                                                            collapse_by_average=is_single_action)
        end = time.time()
        logging.debug(f'Stage 3 [Analyze]: Elapsed time {end - start}')

        # Normalize the outputs
        start = time.time()
        state_analyzed_norm = CompositeActionExtractor._normalize(state_analyzed)
        action_analyzed_norm = CompositeActionExtractor._normalize(action_analyzed, norm_per_dim=not is_single_action)
        end = time.time()
        logging.debug(f'Stage 4: [Normalize] Elapsed time {end - start}')

        # Get binary distributions
        start = time.time()
        state_analyzed_binary = CompositeActionExtractor._get_binary_dist(state_analyzed_norm, keep_percent)
        action_analyzed_binary = CompositeActionExtractor._get_binary_dist(action_analyzed_norm, keep_percent,
                                                                           per_dim=True and not is_single_action)
        end = time.time()
        logging.debug(f'Stage 5 [Get Binary]: Elapsed time {end - start}')

        # Get segments (Hurray)
        start = time.time()
        composite_actions, index_groups = CompositeActionExtractor._get_composite_action_groups(actions,
                                                                                                action_analyzed_binary,
                                                                                                state_analyzed_binary,
                                                                                                window_size)
        end = time.time()
        logging.debug(f'Stage 6 [Get Composite]: Elapsed time {end - start}')

        if k != -1:
            scored_composite_actions = {
                key: [(seg, state_analyzed_norm[~np.isnan(seg)].sum()) for seg in composite_actions[key]] for key in
                composite_actions
            }
            for key in scored_composite_actions:
                scored_composite_actions[key] = list(
                    sorted(scored_composite_actions[key], key=lambda x: x[1], reverse=True))[:k]
        else:
            scored_composite_actions = None

        info = {'state_bw': state_bw, 'actions_bw': actions_bw, 'state_analyzed': state_analyzed,
                'action_analyzed': action_analyzed, 'state_analyzed_norm': state_analyzed_norm,
                'action_analyzed_norm': action_analyzed_norm, 'state_analyzed_binary': state_analyzed_binary,
                'action_analyzed_binary': action_analyzed_binary, 'index_groups': index_groups, 'state': state,
                'actions': actions, 'scored_composite_actions': scored_composite_actions}

        return composite_actions, info

    @staticmethod
    def _get_composite_action_groups(actions: np.array, binary_actions: np.array, state: np.array, window_size):
        l_w = window_size // 2 + 1
        r_w = window_size // 2

        composite_action_dict = {}
        group_dict = {}
        for d in range(binary_actions.shape[1]):
            valid_groups = CompositeActionExtractor._get_valid_segments(binary_actions[:, d], state, l_w, r_w)
            composite_actions = CompositeActionExtractor._get_single_composite_actions(actions, valid_groups)
            composite_action_dict[d] = composite_actions
            group_dict[d] = valid_groups
        return composite_action_dict, group_dict

    @staticmethod
    def _get_single_composite_actions(actions: np.array, valid_groups: np.array):
        actions, valid_groups = actions.copy(), valid_groups.copy()

        sequences = []
        if np.sum(valid_groups) == 1:
            valid_groups[:np.argmax(valid_groups)] = True
            local_action_values = np.copy(actions).copy()
            local_action_values[np.invert(valid_groups)] = None
            sequences.append(local_action_values)
        else:
            indices = np.where(valid_groups)[0]
            for i, index in enumerate(indices[:-1]):
                local_action_values = np.copy(actions).copy()
                local_action_values[:index] = None
                local_action_values[indices[i + 1]:] = None
                sequences.append(local_action_values)
        return sequences

    @staticmethod
    def _get_valid_segments(actions: np.array, state: np.array, l_w, r_w):
        s = [(True in actions[i - l_w:i + r_w] and True in state[i - l_w:i + r_w]) for i in range(state.shape[0])]
        return CompositeActionExtractor._get_bin_group(s)

    @staticmethod
    def _get_binary_dist(series: np.array, threshold, per_dim=False):
        if per_dim:
            return np.apply_along_axis(CompositeActionExtractor._get_binary_dist, 1, series.copy(), threshold)
        return series >= threshold

    @staticmethod
    def _get_bin_group(series: np.array):
        skip = False
        new_sequence = [False] * len(series)
        for i, element in enumerate(series):
            if element and not skip:
                skip = True
            if not element and skip:
                new_sequence[i - 1] = True
                skip = False

        return np.array(new_sequence)

    @staticmethod
    def _analyze(series: np.array, method: str, collapse_by_average: bool):
        """
        Based on the method chose, return the result from that method.

        Based on collapse_by_average, the final array will have its dimension axis averaged so that an input series of
        N x D will be N x 1. For example, state space analysis will be by default averaged, however action analysis will
        returns an array in the original dimension.

        The array will be converted into probabilities of the method requires it. Based on this knowledge, you have the
        option to toggle p_per_dim to determine if the probability should be calculated relative to the current
        dimension, or relative to the entire series.

        Args:
            series:
            method:
            collapse_by_average:

        Returns:
        """
        analyzed_series = None
        registered_names = CompositeActionExtractor.registered_method_names
        chosen_method = [method == _ or method == registered_names[_] for _ in registered_names]

        if any(chosen_method):
            method_name = list(registered_names.keys())[int(np.argmax(chosen_method))]
            needs_q = method_name in list(registered_names.keys())[-4:]
            method = CompositeActionExtractor.registered_method[method_name]

            # Handle method arguments
            if method_name == 'approximate':
                method = partial(method, m=3, r=0)
            if method_name == 'sample':
                method = partial(method, m=2, r=1)

            # Iterate through the dimensions
            for d in range(series.shape[1]):
                if needs_q:
                    # s_d = np.expand_dims(np.apply_along_axis(compositeActionExtractor.get_q, 1, series[:, d], method), 1)
                    s_d = CompositeActionExtractor._analyze_with_q(series[:, d], method)
                else:
                    s_d = np.expand_dims(np.apply_along_axis(method, 1, series[:, d]), 1)
                analyzed_series = np.hstack((analyzed_series, s_d)) if analyzed_series is not None else s_d
        else:
            raise ValueError(f'Method name {method} is invalid. Can be any of the following: {registered_names}')

        return np.average(analyzed_series, axis=1).reshape(-1, 1) if collapse_by_average else analyzed_series

    @staticmethod
    def _analyze_with_q(series: np.array, method):
        """
        Generates a series using the passed in method.

        It will keep the smallest value for each p. For example:
        >>> a = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        If p is:
        >>> p = a[0:5]
        Then is there another pattern that is similar?
        >>> a[1:6]
        Will be a completely different pattern, however from simply looking at the series, it is obvious that there is
        a pattern of [1, 1, 0]. So we get the similarity value for a[1:6], a[2:7], a[3:7], a[4:8]. a[3:7] will
        render the smallest similarity, and is the most similar 'q' to 'p'. In fact, a[3:7] will be 0 meaning that
        we found a complete match!

        Args:
            series:
            method:

        Returns:

        """
        s_d_q = []
        for i in range(series.shape[0]):
            # Once we reach the end of the sequence, we look back instead of forward to avoid out-of-index
            direction = 1 if i < series.shape[0] - 1 - series.shape[1] else -1
            # For each p, the q is the sequence adjacent, or a few steps away from p.
            # We keep the minimum distance, which if 0 will mean there is a constant pattern
            s_d_q.append(min([method(series[i], series[i + direction * j]) for j in range(1, series.shape[1])]))

        return CompositeActionExtractor._remove_nans_negatives(np.array(s_d_q)).reshape(-1, 1)

    @staticmethod
    def _remove_nans_negatives(series: np.array):
        series[np.isnan(series)] = 0
        series[series <= 0] = 0
        return series

    @staticmethod
    def _normalize(series: np.array, norm_per_dim=False):
        if norm_per_dim:
            return np.apply_over_axes(CompositeActionExtractor._normalize, series, 0)
        return (series - series.min()) / (series.max() - series.min())

    @staticmethod
    def _get_p(series: np.array, p_per_dim: bool):
        """
        Gets the probabilities for a series.

        We have the option based on p_per_dim to calculate the probabilities per dimension or over the entire series.
        This is important if we are calculating the probabilities over action dimensions (where we might want to do
        p_per_dim) or over a state space (where we might want to calculate the probabilities with regard to the entire
        state sequence.)

        Args:
            p_per_dim:
            series:

        Returns:
        """
        if not p_per_dim:
            freq_dict = Counter(series.flatten())
            probabilities = dict((k, val / np.prod(series.shape)) for k, val in freq_dict.items())
            return np.vectorize(probabilities.get)(series)
        else:
            # https://stackoverflow.com/questions/48457469/python-convert-a-vector-into-probabilities
            series_p = None
            for d in range(series.shape[1]):
                freq_dict = Counter(series[:, d].flatten())
                probabilities = dict((k, val / np.prod(series[:, d].shape)) for k, val in freq_dict.items())
                series_p_slice = np.vectorize(probabilities.get)(series[:, d]).reshape(-1, 1)
                series_p = series_p_slice if series_p is None else np.hstack((series_p, series_p_slice))
            return series_p

    @staticmethod
    def _bin_p_window(series: np.array, window_size: int, bins: int, p_per_dim):
        """
        Converts a series toa binned series, then as probabilities, and finally as a windowed series.

        Args:
            p_per_dim:
            series:
            window_size:
            bins:

        Returns:

        """
        series_b = np.apply_along_axis(partial(bin_numpy, bins=bins), 0, series)
        series_p = CompositeActionExtractor._get_p(series_b, p_per_dim)
        series_w = CompositeActionExtractor._get_windowed_series(series_p, window_size)
        return series_w

    @staticmethod
    def _get_windowed_series(series: np.array, window_size: int):
        """
        Returns a series in a windowed format.

        Where the windowed series is:

        N x W x D

        Where N is the number of time steps, W is the window_size, and D is the number of dimensions.

        Notes:
            Anyone is welcome to clean up this function. I tried making this a 3 line function, however the dimensions
            for coming out incorrectly. Mainly given a series 200 x 4 (N x D), a single apply along outputs an array
            of 200 x 5 x 4 x 4 which does not make too much sense.

        Args:
            series:
            window_size:

        Returns:

        """
        windowed_series = None
        for d in range(series.shape[1]):
            # You might be able to use apply_over_axes instead
            temp = np.apply_along_axis(CompositeActionExtractor._slide_window, 0, series[:, d], window_size)
            if windowed_series is None:
                windowed_series = np.expand_dims(series[:, d][temp], axis=1)
            else:
                windowed_series = np.hstack((windowed_series, np.expand_dims(series[:, d][temp], axis=1)))
        return windowed_series

    @staticmethod
    def _slide_window(series: np.array, window_size: int):
        """
        Returns a windowed series that is padded via duplicating the end indices.

        Args:
            series:
            window_size:

        Returns:

        """
        left_w = window_size // 2 + 1
        right_w = window_size // 2
        unpadded = np.array([np.arange(i - left_w, i + right_w) for i in range(left_w, series.shape[0] - right_w)])
        return np.vstack((unpadded[:left_w], unpadded, unpadded[-right_w:]))
