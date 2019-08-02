import io
from copy import copy
from pprint import pprint

import numpy as np
from typing import List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm
from composite_action_extractor.utils.misc import bcolors


class PlotObject(object):
    TYPE_VIDEO = 1
    TYPE_REGULAR = 2
    TYPE_SEGMENTED_REGULAR = 3
    TYPE_REGULAR_MULTI_DIMENSIONAL = 8
    META_STATE_ENTROPY = 4
    META_ACTION_ENTROPY = 5
    META_ACTION = 6
    META_STATE = 7
    EMPTY = -1

    def __init__(self, plot_type: int, values: np.array, name: str,
                 meta_type: int = EMPTY, pair_with_type: list = None, soft_name_pair: list = None, display_single=True):
        """
        Each plot object will have a type, values,
        and a name. Depending on its type, it will be handled
        differently.

        Args:
            plot_type (int): Can be video, regular, segmented_regular
            values: A numpy array of the values
            name: The name for the plot to be labeled.
            meta_type (int): Used for over laying plots. Can be state, entropy.
            pair_with_type (list): List of meta_types to overlay onto this object.
            soft_name_pair (list): List of strings for only pair. The list needs to be the same size as the pair with,
                                   however because the name will be correlated with that pair object. If you do not
                                   want to correlate all names for all pairs, just put None.
            display_single (bool): Determines whether to also display this as its own plot.
        """
        self.plot_type = plot_type
        self.values = values
        try:
            if len(self.values) > 0:
                self.values = np.array(values).astype(np.float64) if type(values[0]) is not np.ndarray else values
        except ValueError:
            print('hi')
        self.name = name
        self.meta_type = meta_type
        self.pair_with_type = pair_with_type
        self.soft_name_pair = soft_name_pair
        self.display_single = display_single

        if self.pair_with_type and len(self.soft_name_pair) > 0:
            assert len(self.pair_with_type) == len(self.soft_name_pair), \
                f'Pair_with {len(self.pair_with_type)} and soft_name {len(self.soft_name_pair)} do not match'


class MatplotHandler(object):

    COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

    @staticmethod
    def axes_update_segmented_regular_plot(obj: PlotObject, axes: Axes = None, obj_list: List[PlotObject] = None,
                                           step=None, n_frames=None):
        """
        Segmented means this is technically to be treated as a single line,
        however needs a legend and colored segments.

        Args:
            obj:
            axes:
            obj_list:
            step:

        Returns:

        """
        _ = obj_list
        x_increment = 0
        marker_y_value = 0

        n_values = len([item for sublist in obj.values for item in sublist])
        step = round(step * n_values / n_frames) if round(step * n_values / n_frames) < n_values else n_values - 1
        step = step if step != -1 else 0

        is_added = False
        for i, sequence in enumerate(obj.values):
            in_loop_sequence = list(sequence)
            in_loop_sequence += [obj.values[i + 1][0]] if i != len(obj.values) - 1 else [obj.values[i][-1]]
            if i < 5:
                axes.plot(np.arange(x_increment, x_increment + len(in_loop_sequence)), in_loop_sequence,
                          label=f'composite Action {obj.name} {i}')
            else:
                axes.plot(np.arange(x_increment, x_increment + len(in_loop_sequence)), in_loop_sequence)

            x_increment += len(sequence)
            if not is_added and step < x_increment:
                marker_y_value = obj.values[i][step - (x_increment - len(sequence))]
                is_added = True

        marker_plot = [None] * x_increment
        # print(f'step {step} marker size {len(marker_plot)} x inc {x_increment}')

        if step < x_increment:
            #print(f'Marker plot size {len(marker_plot)} given step {step} where elngth of values are {n_values} and frame size is {n_frames} and values output is {obj.values}')
            marker_plot[step] = marker_y_value
            step = [step]
        else:
            step = None
        axes.plot(np.arange(0, len(marker_plot)), marker_plot, marker='|', markerfacecolor='r', markersize=50,
                  markevery=step, markeredgewidth=2, markeredgecolor='r')
        axes.legend()
        axes.set_title(obj.name)

    @staticmethod
    def axes_update_image_plot(obj: PlotObject, axes: Axes = None, obj_list: List[PlotObject] = None, step=None, n_frames=None):
        """
        Updates the axis with an image. If the Plot object contains multiple frames, then it will update
        the axes with the frame at the step number.

        Args:
            obj:
            axes:
            obj_list:
            step:

        Returns:

        """
        _ = obj_list
        axes.set_title('Video')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.grid(False)

        # Check if the image is grey scale, and add the image to matplotlib
        if obj.values[step].shape[0] == 1:
            axes.imshow(obj.values[step].squeeze(0), cmap=plt.cm.binary)
        else:
            # try:
            axes.imshow(obj.values[step])
            # except Exception as e:
            #     print(f'Frame crash {e}')

    @staticmethod
    def axes_update_regular_plot(obj: PlotObject, axes: Axes = None, obj_list: List[PlotObject] = None, step=None, n_frames=None):
        """
        Regular line plot.

        Args:
            obj:
            axes:
            obj_list:
            step:

        Returns:

        """
        axes.set_title(f'{obj.name}')
        axes.grid(False)

        if np.isscalar(obj.values[0]):
            n_values = len(obj.values)
        else:
            try:
                n_values = len([item for sublist in obj.values for item in sublist])
            except TypeError:
                print('hi')

        markers_on = None
        if step is not None:
            step = round(step * n_values / n_frames) if round(step * n_values / n_frames) < n_values else n_values - 1
            markers_on = [step] if step < n_values else None
        color_iter = 0

        denominator = np.max(obj.values) - np.min(obj.values)
        denominator = denominator if denominator != 0 else 1
        # print(f'Dividing: values by min {np.min(obj.values)} {denominator} min {np.min(obj.values)} max {np.max(obj.values)}')
        if markers_on:
            axes.plot(np.divide(obj.values - np.min(obj.values), denominator),
                      marker='|', markerfacecolor='r', markersize=50, markevery=markers_on, markeredgewidth=2,
                      markeredgecolor='r', color=MatplotHandler.COLORS[color_iter], label=obj.name, zorder=1)
        else:
            axes.plot(np.divide(obj.values - np.min(obj.values), denominator),
                      color=MatplotHandler.COLORS[color_iter], label=obj.name, zorder=1)
        # Add any other plots we want to associate this object with based on the pair with type
        for other_plot in [_ for _ in obj_list if obj.pair_with_type and _.meta_type in obj.pair_with_type]:
            if obj.soft_name_pair and other_plot.name != obj.name:
                # If there is a name pair for a pairing type, then handle it differently.
                common_key = obj.soft_name_pair[obj.pair_with_type.index(other_plot.meta_type)]
                if common_key and other_plot.name.__contains__(common_key):
                    color_iter += 1
                    denominator = np.max(other_plot.values) - np.min(other_plot.values)
                    denominator = denominator if denominator != 0 else 1
                    axes.plot(np.divide(other_plot.values - np.min(other_plot.values), denominator),
                              label=other_plot.name, color=MatplotHandler.COLORS[color_iter], zorder=1)
                # Alternatively, if there is not name key, then proceed to plot the axis
                elif not common_key and other_plot.name != obj.name:
                    other_axis_max = np.sum(np.abs([np.max(other_plot.values), np.min(other_plot.values)]))
                    main_axis_max = np.sum(np.abs([np.max(obj.values), np.min(obj.values)]))
                    color_iter += 1
                    if abs(other_axis_max - main_axis_max) < 0.1:
                        axes.plot(other_plot.values, label=other_plot.name, color=MatplotHandler.COLORS[color_iter],
                                  zorder=0, alpha=0.5)
                    else:
                        ax2 = axes.twinx()
                        ax2.plot(other_plot.values, label=other_plot.name, color=MatplotHandler.COLORS[color_iter],
                                 zorder=0, alpha=0.5)
                        ax2.set_ylabel(other_plot.name)
        if obj.pair_with_type:
            axes.legend()

    @staticmethod
    def get_figure_image(obj_list: List[PlotObject], n_rows=0, n_cols=4, step=None, n_frames=None, figsize=(6, 5)):
        n_single_plots = len([obj for obj in obj_list if obj.display_single])
        n_rows = n_rows if n_rows > 0 else np.ceil(n_single_plots / n_cols).astype(int)
        n_cols = n_cols if 0 < n_cols <= n_single_plots else n_single_plots

        figure = plt.figure(figsize=(figsize[0] * n_cols, figsize[1] * n_rows))  # type: Figure
        sub_axis = []

        # Load the axis plots
        for i, obj in enumerate([obj for obj in obj_list if obj.display_single]):
            sub_axis.append(figure.add_subplot(n_rows, n_cols, i + 1))  # type: List[Axes]
            if obj.plot_type == obj.TYPE_VIDEO:
                MatplotHandler.axes_update_image_plot(obj, sub_axis[-1], obj_list, step, n_frames)
            elif obj.plot_type == obj.TYPE_REGULAR:
                MatplotHandler.axes_update_regular_plot(obj, sub_axis[-1], obj_list, step, n_frames)
            elif obj.plot_type == obj.TYPE_SEGMENTED_REGULAR:
                MatplotHandler.axes_update_segmented_regular_plot(obj, sub_axis[-1], obj_list, step, n_frames)
            else:
                print(f'\n{bcolors.WARNING} We have an object that has a strange plot_type. Dumping...\n')
                pprint(vars(obj))
                print(bcolors.ENDC)

        figure.subplots_adjust(wspace=0.4)
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Create Image object
        return np.array(Image.open(buf))[:, :, :3]

    @staticmethod
    def get_plot_sequence(obj_list: List[PlotObject], n_rows=0, n_cols=4):
        frames = []
        if PlotObject.TYPE_VIDEO in [_.plot_type for _ in obj_list]:
            n_frames = [_ for _ in obj_list if _.plot_type == PlotObject.TYPE_VIDEO][-1].values.shape[0]
            for i in tqdm(range(n_frames)):
                frames.append(MatplotHandler.get_figure_image(obj_list, n_rows, n_cols, i, n_frames))
        else:
            frames.append(MatplotHandler.get_figure_image(obj_list, n_rows, n_cols, 0))
        return frames
