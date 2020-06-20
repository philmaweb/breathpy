from io import BytesIO
import graphviz
import matplotlib

matplotlib.use('Agg')
import os
import shutil
import tempfile

from decimal import Decimal
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy import interp
from skimage.transform import resize
from skimage import color as skcolor
from sklearn.metrics import auc


from ..model.BreathCore import PeakAlignmentMethod, MccImsAnalysis, MccImsMeasurement
from ..model.BreathCore import deprecated


# Collection of plotting functions
class HeatmapPlot(object):
    """
    Collection of Heatmap plots
    """


    @staticmethod
    def prepare_fast_heatmap_plot(df, cmap_dict):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # faster conversion
        image = skcolor.rgb2gray(df.T.values)
        # image.shape
        x_size, y_size = 2500, 2500
        image_resized = resize(image, (x_size, y_size), mode='reflect')
        plt.imshow(image_resized, **cmap_dict)
        ax.invert_yaxis()
        plt.colorbar()

        # to plot we need to use the indices of the intensity matrix - not the actual values
        irm_vals = np.asarray(df.index.values, dtype=float)
        rt_vals = np.asarray(df.T.index.values, dtype=float)
        # need to re-adjust the stepwidth - as we resized the image
        ClusterPlot.setup_axes_for_fast_intensity_matrix(ax, irm_vals=irm_vals, rt_vals=rt_vals, x_size=x_size,
                                                         y_size=y_size)
        fig.set_figsize = (9, 4)
        return fig, ax


    @staticmethod
    def plot_heatmap_helper_fast(mcc_ims_measurement, plot_parameters, title="", title_prefix="Measurement ",
                                 plot_dir_suffix="heatmaps", plot_type="intentsity_plot"):
        intensity_matrix = mcc_ims_measurement.df
        measurement_name = mcc_ims_measurement.filename
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # # faster conversion
        # image = skcolor.rgb2gray(intensity_matrix.T.values)
        # # image.shape
        # x_size, y_size = 2500, 2500
        # image_resized = resize(image, (x_size, y_size), mode='reflect')
        # plt.imshow(image_resized, cmap=plot_parameters['colormap'], vmin=0, vmax=1)
        # ax.invert_yaxis()
        # plt.colorbar()
        #
        # # to plot we need to use the indices of the intensity matrix - not the actual values
        # irm_vals = np.asarray(intensity_matrix.index.values, dtype=float)
        # rt_vals = np.asarray(intensity_matrix.T.index.values, dtype=float)
        # # need to re-adjust the stepwidth - as we resized the image
        # ClusterPlot.setup_axes_for_fast_intensity_matrix(ax, irm_vals=irm_vals, rt_vals=rt_vals, x_size=x_size, y_size=y_size)
        # fig.set_figsize = (9, 4)
        #
        fig, ax = HeatmapPlot.prepare_fast_heatmap_plot(intensity_matrix, {"cmap":plot_parameters['colormap'], "vmin":0, "vmax":1})
        if not title:
            title_string = '{}{}'.format(title_prefix, measurement_name)
        else:
            title_string = title

        plt.yticks(rotation=0)
        ax.set_title(title_string)

        figure_dir = '{}{}/'.format(plot_parameters['plot_dir'], plot_dir_suffix)

        plot_prefix = ''
        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']
            plot_prefix = "fast_{}".format(plot_prefix)
            # import pdb;pdb.set_trace()
        dataset_name = plot_prefix

        figure_name = '{}_{}_{}.png'.format(dataset_name, plot_type, measurement_name[:-4])

        if not plot_parameters.get('use_buffer', False):
            Path(figure_dir).mkdir(parents=True, exist_ok=True)
            print(f"Saving figure to {figure_dir}{figure_name}")
            fig.savefig("{}{}".format(figure_dir, figure_name), dpi=200, bbox_inches='tight', format="png",
                        compress_level=1)
        return_figure = save_plot_to_buffer(plot_parameters, fig)
        plt.close()
        return return_figure


    @staticmethod
    def plot_classwise_heatmaps(mccImsAnalysis, plot_parameters):
        """
        Group Measurements by their class label and create an averaged heatmap
        :type mccImsAnalysis: MccImsAnalysis or list(model.BreathCore.MccImsMeasurement) with assigned class_labels
        :param mccImsAnalysis: MccImsAnalysis holding the class labels and measurements
        :param plot_parameters: dictionary holding plotting parameters
        :return: [tuple(Buffer,....) / None], saves consensus heatmaps
        """
        plots_to_return = []
        if isinstance(mccImsAnalysis, MccImsAnalysis):
            class_labels = np.unique([m.class_label for m in mccImsAnalysis.measurements])
            class_label_measurement_dict = {cl: [] for cl in class_labels}
            for m in mccImsAnalysis.measurements:
                class_label_measurement_dict[m.class_label].append(m)

        elif isinstance(mccImsAnalysis, list):
            class_labels = np.unique([m.class_label for m in mccImsAnalysis])
            class_label_measurement_dict = {cl: [] for cl in class_labels}
            for m in mccImsAnalysis:
                class_label_measurement_dict[m.class_label].append(m)
        else:
            raise ValueError("Expected MccImsAnalysis or list of measurements, not {}".format(type(mccImsAnalysis)))

        print(class_label_measurement_dict)
        for k,v in class_label_measurement_dict.items():
            plots_to_return.append(HeatmapPlot._plot_classwise_heatmap_helper(v, k, plot_parameters))
        return plots_to_return
    
    @staticmethod
    def compute_averaged_df(dfs):
        averaged_df = pd.DataFrame()
        # not very simple, need to do index things - need to find fewest rows and fewest columns, then crop all of them to same dimensions
        # then drop indices and add to each other
        # then assign original index
        # is_measurement_list = isinstance(dfs[0], MccImsMeasurement)
        if isinstance(dfs[0], pd.DataFrame):
            # real_indices = [df.index for df in dfs]
            lowest_x = max(0.0, min([df.index.min() for df in dfs]))
            lowest_highest_x = min([df.index.max() for df in dfs])
            lowest_y = max(0.0, min([df.T.index.min() for df in dfs]))
            lowest_highest_y = max(0.0, min([df.T.index.max() for df in dfs]))
        else:
            raise ValueError("Expected list of MccImsMeasurements or pandas.DataFrames, not [{}]".format(
                type(dfs[0])))

        averaged_index, transposed_average_index = None, None
        for i, df in enumerate(dfs):
            # if is_measurement_list:
            selected_df = df[np.logical_and(df.index >= lowest_x, df.index <= lowest_highest_x)]
            selected_df_transp = selected_df.T
            selected_df_transp = selected_df_transp[
                np.logical_and(selected_df_transp.index >= lowest_y, selected_df_transp.index <= lowest_highest_y)]

            # save first index for later reassignment
            if not i:
                averaged_index = selected_df.index.copy()
                transposed_average_index = selected_df_transp.index.copy()

            selected_df_transp.reset_index(inplace=True, drop=True)
            selected_df = selected_df_transp.T

            # pandas really tries to make adding non-complying shapes / indices hard -
            # so we need to adjust the dimensions of the df even more -> pad with 0.0 columns / rows to the end
            # or delete last column/row to fit shape
            # print(selected_df.shape)
            selected_df.reset_index(inplace=True, drop=True)
            # print(selected_df.index)
            # print(selected_df.T.index)
            if i:
                if average_shape[0] > selected_df.shape[0]:
                    selected_df[len(selected_df)] = 0.0
                elif average_shape[0] < selected_df.shape[0]:
                    appropriate_len = average_shape[0] - 1
                    selected_df = selected_df[selected_df.index <= appropriate_len]

                if average_shape[1] > selected_df.shape[1]:
                    selected_df.T[average_shape[1]] = 0.0
                elif average_shape[1] < selected_df.shape[1]:
                    appropriate_len = average_shape[1] - 1
                    selected_df = selected_df.T[selected_df.T.index <= appropriate_len]
                    selected_df = selected_df.T

                averaged_df += selected_df
            else:
                averaged_df = selected_df
            average_shape = averaged_df.shape

        averaged_df /= len(dfs)

        # but we need an index for our heatmap plot --> reassign index with index of first element
        averaged_df.set_index(averaged_index, drop=True, inplace=True)
        averaged_df_t = averaged_df.T
        averaged_df_t.set_index(transposed_average_index, drop=True, inplace=True)
        return averaged_df_t.T

    @staticmethod
    def _plot_classwise_heatmap_helper(mccImsMeasurements, class_label, plot_parameters):
        """
        Compute average for grouped Measurements and return buffer or None
        :type mccImsMeasurements: [MccImsMeasurements,...]
        :param mccImsMeasurements: mccImsMeasurements each with the same class label
        :param plot_parameters: dictionary holding plotting parameters
        :return: tuple(Buffer,....) / None, saves averaged heatmaps
        """
        
        averaged_df = HeatmapPlot.compute_averaged_df([m.df for m in mccImsMeasurements])
        
        dummy_measurement = MccImsMeasurement("{} ".format(class_label), init_df=averaged_df, class_label=class_label)
        descriptor = "Averaged_class {}".format(class_label)
        return (class_label, HeatmapPlot.plot_heatmap_helper_fast(mcc_ims_measurement=dummy_measurement, plot_parameters=plot_parameters, title=descriptor, plot_type="classwise_heatmap",))

    @staticmethod
    def plot_feature_matrix(feature_matrix, class_label_dict, plot_parameters, outname=""):
        """
        TODO needs annotation
        Visualize the feature matrix by class using the normalized intensity of each feature
        :param feature_matrix:
        :param class_label_dict:
        :param plot_parameters:
        :param outname:
        :return:
        """
        from collections import defaultdict
        fig = plt.figure()
        # fig = ax.get_figure()
        # iterate over all class labels - filter rows by correct label and create subplot
        class_to_row_map = defaultdict(list)
        for rn, cl in class_label_dict.items():
            class_to_row_map[cl].append(rn)

        num_loops = len(class_to_row_map.keys())
        for i, (cl, row_selection) in enumerate(class_to_row_map.items()):
            # create mask for selection of correct rows
            selection_mask = [i in row_selection for i in feature_matrix.index]
            relevant_rows = feature_matrix.loc[selection_mask]

            ax = fig.add_subplot(2,1,i+1)
            # force all x labels...
            sns.heatmap(relevant_rows, ax=ax, xticklabels=True)
            ax.set_ylabel(cl)
            # remove x-axis label for all axes except the last one
            if i < num_loops-1: # enumerate starts at 0 and will end at len(-1)
                ax.get_xaxis().set_ticks([])
            # else:
            #     ax.get_xaxis().set_ticks(relevant_rows.columns.values)

        # fig, ax = HeatmapPlot.prepare_fast_heatmap_plot(feature_matrix,
        #                                                 {"cmap": plot_parameters['colormap'], "vmin": 0, "vmax": 1})

        # plt.yticks(rotation=0)
        if not outname:
            plot_suffix = "fm_plot",
            figure_dir = f"{plot_parameters['plot_dir']}{plot_suffix}"
            figure_name = f'{plot_suffix}.png'
        else:
            figure_dir = Path(outname).parent
            figure_name= Path(outname).stem + Path(outname).suffix

        # make subplot for all rows of certain class
        if not plot_parameters.get('use_buffer', False):
            Path(figure_dir).mkdir(parents=True, exist_ok=True)
            print(f"Saving figure to {figure_dir}/{figure_name}")
            fig.savefig(f"{figure_dir}/{figure_name}", dpi=300, bbox_inches='tight', format="png",
                        compress_level=1)
        return_figure = save_plot_to_buffer(plot_parameters, fig)
        plt.close()
        return return_figure


    # IntensityMatrix = _plot_heatmap_from_mcc_ims_measurement
    # BasicHeatmap = plot_heatmap_helper
    FastIntensityMatrix = plot_heatmap_helper_fast
    ClasswiseHeatmaps = plot_classwise_heatmaps
    FeatureMatrixPlot = plot_feature_matrix
    # MultipleIntensityMatrices = plot_intensity_matrices


class ClusterPlot(object):
    """
    Collection of Cluster plots
    """
    @staticmethod
    def _map_clustering_parameters_to_string(param_dict, method_name):
        """
        Takes clustering parameters and constructs them to a String
        :param param_dict:
        :param method_name:
        :return:
        """
        method_parameter_str = ""
        # map method names to parameter names to include them in plot for comparison
        # some clustering methods can have multiple parameters
        map_method_parameter = {PeakAlignmentMethod.AFFINITY_PROPAGATION_CLUSTERING.name: "preference",
                                PeakAlignmentMethod.MEAN_SHIFT_CLUSTERING.name: "bandwidth",
                                PeakAlignmentMethod.K_MEANS_CLUSTERING.name: "n_clusters",
                                PeakAlignmentMethod.DB_SCAN_CLUSTERING.name: "eps",
                                PeakAlignmentMethod.WINDOW_FRAME.name: "distance_threshold",
                                PeakAlignmentMethod.PROBE_CLUSTERING.name: "threshold_inverse_reduced_mobility",
                                # threshold_inverse_reduced_mobility = threshold_inverse_reduced_mobility,
                                # threshold_scaling_retention_time = threshold_scaling_retention_time
                                }
        method_parameter = param_dict.get(map_method_parameter[method_name], '')
        method_parameter_name = map_method_parameter[method_name]

        if param_dict:
            if isinstance(method_parameter, float):
                method_parameter_str = "{:.4f}".format(method_parameter)
            elif isinstance(method_parameter, tuple):
                method_parameter_str = " ".join("{:.4f}").format(*method_parameter)
            else:
                method_parameter_str = str(method_parameter)
            return "{} = {}".format(method_parameter_name, method_parameter_str)
        return method_parameter_str

    @staticmethod
    def _plot_clustering(intermediate_analysis_result, plot_parameters):
        """
        Plot Peak alignment result 1 png per peak_detection method, using scatterplot
        :param intermediate_analysis_result: MccImsAnalysis object
        :param plot_parameters:
        :return:
        """
        figure_dir = f"{plot_parameters['plot_dir']}clustering"
        # first check whether directory exists for figure output - or whether we want to use a buffer
        if not plot_parameters.get('use_buffer', False):
            Path(figure_dir).mkdir(parents=True, exist_ok=True)

        plots_to_return = []
        peak_alignment_result = intermediate_analysis_result.peak_alignment_result
        peak_detection_steps = intermediate_analysis_result.peak_detection_combined
        clustering_parameters = peak_alignment_result.params_used_for_clustering
        n_clusters_dict = peak_alignment_result.number_of_clusters

        clustering_method_name = peak_alignment_result.params_used_for_clustering['peak_alignment_step'].name

        for peak_detection in peak_detection_steps:
            # map method names to parameter names to include them in plot for comparison
            method_parameter_str = ClusterPlot._map_clustering_parameters_to_string(param_dict=clustering_parameters,
                                                                                    method_name=clustering_method_name)
            title_string = "Method = {}, {} clusters, {}".format(
                clustering_method_name, n_clusters_dict[peak_detection.name].shape[0], method_parameter_str)

            fig_name = "{}clustering/{}_{}_{}_{}.png".format(
                plot_parameters['plot_dir'], plot_parameters['plot_prefix'], peak_detection.name, clustering_method_name,
                method_parameter_str)
            # x = peak_alignment_result.peak_coordinates[peak_detection.name]['inverse_reduced_mobility']
            # y = peak_alignment_result.peak_coordinates[peak_detection.name]['retention_time']
            # plt.scatter(x, y)
            coordinate_df = peak_alignment_result.peak_coordinates[peak_detection.name]
            # Do we need to use a consistent axis range? - not really
            sns.set_style('whitegrid')
            g = sns.lmplot(data=coordinate_df, x='inverse_reduced_mobility', y='retention_time', fit_reg=False)
            ax = g.axes[0, 0]
            ax.margins(0.05)

            # Compose multiple plots to one pdf file
            # https: // matplotlib.org / faq / howto_faq.html  # save-multiple-plots-to-one-pdf-file
            xlabel = "Inverse Reduced Mobility (1/k0) [Vs/cm^2]"
            ylabel = "Retention Time [s]"

            plt.yticks(rotation=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title_string)

            fig = plt.gcf()
            # distinguish different used methods by using tuples / a hashmap?
            plots_to_return.append((clustering_method_name, peak_detection.name, save_plot_to_buffer(plot_parameters, fig), fig_name))
            if not plot_parameters.get('use_buffer', False):
                fig.savefig(fig_name, dpi=300, bbox_inches='tight', format="png")
            plt.close()

        return plots_to_return

    @staticmethod
    def _plot_multiple_peak_detection_methods_clustering(mcc_ims_analysis, plot_parameters):
        """
        For each Peak alignment results plot 1 png
        :param mcc_ims_analysis: MccImsAnalysis object
        :param plot_parameters:
        :return:
        """
        figure_dir = f"{plot_parameters['plot_dir']}clustering"
        # first check whether directory exists for figure output - or whether we want to use a buffer
        if not plot_parameters.get('use_buffer', False):
            Path(figure_dir).mkdir(parents=True, exist_ok=True)

        peak_alignment_result = mcc_ims_analysis.peak_alignment_result
        peak_detection_steps = mcc_ims_analysis.peak_detection_combined
        clustering_parameters = peak_alignment_result.params_used_for_clustering
        # n_clusters_dict = peak_alignment_result.number_of_clusters

        method = peak_alignment_result.params_used_for_clustering['peak_alignment_step'].name

        method_parameter_str = ClusterPlot._map_clustering_parameters_to_string(param_dict=clustering_parameters,
                                                                                method_name=method)
        title_string = "Method = {}, {}".format(
            method, method_parameter_str)

        fig_name = "{}clustering/{}_{}_{}_{}.png".format(
            plot_parameters['plot_dir'], plot_parameters['plot_prefix'], "multi_peak_detections", method,
            method_parameter_str)

        # assign pdm column for hue
        # join them together in dataframe
        # use lmplot and hue=pdm
        df = pd.concat((peak_coords.assign(peak_detection_method=pdm) for pdm, peak_coords in peak_alignment_result.peak_coordinates.items()))
        sns.set_style('whitegrid')
        g = sns.lmplot(data=df, x='inverse_reduced_mobility', y='retention_time', fit_reg=False,
                       hue='peak_detection_method')
        ax = g.axes[0, 0]
        ax.margins(0.05)

        # Compose multiple plots to one pdf file
        # https: // matplotlib.org / faq / howto_faq.html  # save-multiple-plots-to-one-pdf-file
        xlabel = "Inverse Reduced Mobility (1/k0) [Vs/cm^2]"
        ylabel = "Retention Time [s]"

        plt.yticks(rotation=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_string)

        fig = plt.gcf()
        # distinguish different used methods by using tuples / a hashmap?
        if not plot_parameters.get('use_buffer', False):
            fig.savefig(fig_name, dpi=300, bbox_inches='tight', format="png")
        plot_to_return = save_plot_to_buffer(plot_parameters, fig)
        plt.close()
        return plot_to_return, fig_name

    @staticmethod
    def _compute_stepwidth_reflecting_multiple(vals, multiple):
        stepwidth = np.mean([vals[i+1] - vals[i] for i in range(len(vals) - 1)])
        return (multiple / float(stepwidth))


    @staticmethod
    def compute_stepwidth_fast(vals, target_size, num_points_axis, start_at=0.0):
        if start_at - 0.001 < 0.0:
            return target_size / (vals[-1]/num_points_axis)
        else:
            # print(f"Using cropped adjustment with {start_at}")
            # need to adjust stepwidth - as we potentially scaled a cropped image
            return target_size / ((vals[-1]-start_at) / num_points_axis)

    #
    # @staticmethod
    # def _translate_values_to_indices(index_values, to_scale, use_crop_correction=False):
    #     ids = np.asarray(index_values, dtype=float)
    #     ids_stepwidth = np.mean([ids[i+1] - ids[i] for i in range(len(ids) - 1)])
    #
    #     # get lowest val and scale by stepwidth
    #     lowest_val = np.min(ids)
    #     if use_crop_correction:
    #         scaled_lowest_val = lowest_val // ids_stepwidth + (lowest_val % ids_stepwidth) / ids_stepwidth
    #     else:
    #         scaled_lowest_val = 0.0
    #
    #     # need to assign index - get approx index and use float as relative index to scale
    #     # remove auto cropped out index (by heatmapplot) from calculated index
    #     def scale_to_index(val):
    #         basic_index = val // ids_stepwidth
    #         rest_index = (val % ids_stepwidth) / ids_stepwidth
    #         return (basic_index + rest_index) - scaled_lowest_val
    #
    #     return np.vectorize(scale_to_index)(to_scale)


    @staticmethod
    def _add_scaled_coords_fast(irm_index_values, rt_index_values, coords):
        """
        Scale coords using index values - and add with "scaled_" prefix to coords DataFrame
        :param irm_index_values:
        :param rt_index_values:
        :param coords:
        :return:
        """
        # for irm we need to crop correct -> need to get lowest value from coord axis and subtract the scaled value from
        # all our values, otherwise the coords will be right shifted
        irm_ids = np.asarray(irm_index_values, dtype=float)

        # compute width in pixels that corresponds to a change in 1.0 on the axis


        # irm_to_1_factor = 1/(irm_ids[-1]/2500.0)
        # irm_range = irm_ids[-1] - irm_ids[0]
        min_irm_val = max(0.0, np.min(irm_ids))
        irm_to_1_factor = 1/((irm_ids[-1] - min_irm_val) / 2500.0)

        pixels_to_shift_left = min_irm_val * irm_to_1_factor
        # print(f"Shifting rects by {pixels_to_shift_left} to the left")

        rt_ids = np.asarray(rt_index_values, dtype=float)
        rt_to_1_factor = 1 / (rt_ids[-1] / 2500)
        # rt_to_1_factor pixels is a change in one second retention time

        new_coords = coords.assign(
            scaled_irm=(coords['inverse_reduced_mobility']*irm_to_1_factor) - pixels_to_shift_left,
            scaled_rt=coords['retention_time']*rt_to_1_factor,
            scaled_irm_radius=coords['radius_inverse_reduced_mobility']*irm_to_1_factor,
            scaled_rt_radius=coords['radius_retention_time']*rt_to_1_factor)
        return new_coords



    @staticmethod
    def setup_axes_for_fast_intensity_matrix(ax, irm_vals, rt_vals, y_steps_width=25.0, x_steps_width=0.1,
                                             xlabel="Inverse Reduced Mobility (1/k0) [Vs/cm^2]",
                                             ylabel="Retention Time [s]", x_size=2500, y_size=2500):
        # need to adjust tick positions to multiples of 25 and 0.1 otherwise - its going to use the index of the df
        ax.yaxis.set_major_locator(ticker.MultipleLocator(
            ClusterPlot.compute_stepwidth_fast(rt_vals, y_steps_width, y_size)))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(
            ClusterPlot.compute_stepwidth_fast(irm_vals, x_steps_width, x_size, start_at=irm_vals[0])))

        rt_top_tick = ((np.max(rt_vals) // y_steps_width) + 2.0) * y_steps_width

        irm_top_tick = ((np.max(irm_vals) // x_steps_width) + 2.0) * x_steps_width
        # get one ticklable bellow the actual one, as the first one gets swallowed
        irm_bottom_tick = max(0.0, ((np.min(irm_vals) // x_steps_width) - 1.0) * x_steps_width)

        # need to add 0.0 twice to y-axis as it "swallows" the first 0.0
        y_tick_labels = [0.0]
        for ele in np.arange(0, rt_top_tick, y_steps_width):
            y_tick_labels.append(ele)

        # for x ticks we have another problem, as we crop a significant part of the axis,
        #   our limit is not 0.0, but usually 0.4
        # need to use continous array and add the minimum value
        if irm_bottom_tick == 0.0:
            x_tick_labels = [0.0]
            x_tick_labels.extend(np.arange(0.0, irm_top_tick, x_steps_width))
        else:
            x_tick_labels = np.ascontiguousarray(np.arange(0, irm_top_tick, x_steps_width)) + irm_bottom_tick

        # need to convert to rounded representation
        x_tick_labels = np.round(x_tick_labels, decimals=2)
        # now we assign the "multiple of" labels to the axis - Fixed
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(y_tick_labels))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_tick_labels))
        # reduce font size for labels
        ax.xaxis.set_tick_params(labelsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #  remove black box surrounding the canvas at top and right side
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    @staticmethod
    def _setup_overlay_fast(intensity_matrix, plot_parameters, figure_dir):
        # filename = mcc_ims_measurement.filename
        fig, ax = HeatmapPlot.prepare_fast_heatmap_plot(intensity_matrix, {"cmap":plot_parameters['colormap'], "vmin":0, "vmax":1})

        # to plot we need to use the indices of the intensity matrix - not the actual values
        irm_vals = np.asarray(intensity_matrix.index.values, dtype=float)
        rt_vals = np.asarray(intensity_matrix.T.index.values, dtype=float)

        if not plot_parameters.get('use_buffer', False):
            Path(figure_dir).mkdir(parents=True, exist_ok=True)

        return ax, fig, irm_vals, rt_vals

    @staticmethod
    def _construct_coord_rects(scaled_coords, color):
        # lower left edge is at x,y - so we need to remove radius to match it
        # (x,y), w,h
        rect_coord_tuples = [((row['scaled_irm'] - row['scaled_irm_radius'],
           row['scaled_rt'] - row['scaled_rt_radius']),
          2. * row['scaled_irm_radius'], 2. * row['scaled_rt_radius'],
          )
                for _, row in scaled_coords.iterrows()]
        # print(f"Passed color = {color}")
        return [patches.Rectangle(*coord, fill=False, edgecolor=color, linestyle='dashed', )
                for coord in rect_coord_tuples]


    @staticmethod
    def _plot_overlay_clustering_helper_fast(intensity_matrix, pdm_coord_tuples, plot_parameters, title_dict, best_features_df=None):
        """
        Plot intensity_matrix and detected peaks on top with deviation indicated with rects
        iterating over multiple peak_detection results
        :param intensity_matrix:
        :param pdm_coord_tuples:
        :param plot_parameters:
        :param title_dict:
        :param best_features_df:
        :return:
        """
        return_figures = []
        figure_dir = f"{plot_parameters['plot_dir']}overlay/"
        # first check whether directory exists for figure output - or whether we want to use a buffer
        if not plot_parameters.get('use_buffer', False):
            Path(figure_dir).mkdir(parents=True, exist_ok=True)

        # plot intensity matrix and setup axis
        ax, fig, irm_vals, rt_vals = ClusterPlot._setup_overlay_fast(intensity_matrix, plot_parameters, figure_dir)

        if best_features_df is not None:
            # CASE1 - Best Overlay Plot
            for pdmn, peak_coords_original in pdm_coord_tuples:
                title_dict['peak_detection_method_name'] = pdmn
                best_peaks_by_pdm = best_features_df[best_features_df['peak_detection_method_name'] == pdmn]

                # scaled_coords_original = ClusterPlot._add_scaled_coords(irm_vals, rt_vals, peak_coords_original)

                # now iterate over different Feature reduction methods
                for performance_measure_name, best_peaks_by_pdm_pm in best_peaks_by_pdm.groupby("performance_measure_name"):
                    title_dict['performance_measure'] = performance_measure_name
                    # TODO - also group reduced features by class comparison? - so far only binary support
                    best_peak_coords = best_peaks_by_pdm_pm[peak_coords_original.columns.values]
                    scaled_best_coords = ClusterPlot._add_scaled_coords_fast(irm_vals, rt_vals, best_peak_coords)
                    # print(scaled_coords_original)
                    # print(scaled_best_coords)
                    # TO check if we have identical peak_coords, call this:
                    # print(scaled_coords_original.loc[scaled_coords_original['peak_id'].isin(scaled_best_coords.peak_id.values)].reset_index(inplace=True) == scaled_best_coords.reset_index(inplace=True))
                    # rects_orig = ClusterPlot._construct_coord_rects(scaled_coords_original, color="red")
                    rects_best = ClusterPlot._construct_coord_rects(scaled_best_coords, color="blue")
                    added_rects = [ax.add_patch(rect) for rect in rects_best]

                    title_string = f"Best Peaks in {title_dict['measurement_name']},\n {title_dict['peak_detection_method_name']}, {title_dict['performance_measure']}"

                    ax.set_title(title_string)
                    prefix = ""
                    if title_dict['dataset_name']:
                        prefix = f"{title_dict['dataset_name']}_"
                    figure_name = f"{prefix}best_overlay_{title_dict['peak_detection_method_name']}_{title_dict['measurement_name']}_{title_dict['performance_measure']}.png"

                    if not plot_parameters.get('use_buffer', False):
                        fig.savefig(f"{figure_dir}{figure_name}", dpi=300, bbox_inches='tight', format="png")
                    return_figures.append(((pdmn, title_dict['measurement_name'], performance_measure_name), save_plot_to_buffer(plot_parameters, fig), figure_name))
                    # remove rects for next iteration
                    for rect in added_rects: rect.remove()

        else:
            # CASE2 - Overlay Plot by PeakDetection
            # dont plot best features - only peak detection
            for pdmn, peak_coords in pdm_coord_tuples:
                title_dict['peak_detection_method_name'] = pdmn
                # are the coords correct?
                scaled_coords = ClusterPlot._add_scaled_coords_fast(irm_vals, rt_vals, peak_coords)

                rects = ClusterPlot._construct_coord_rects(scaled_coords, color="black")
                # cluster_centers = [(row['inverse_reduced_mobility'], row['retention_time']) for _, row in new_scaled_coords.iterrows()]
                added_rects = [ax.add_patch(rect) for rect in rects]

                # Scatterplot to mark peak positions
                # ax.scatter(data=scaled_coords, x="scaled_irm", y="scaled_rt", marker='*')
                title_string = f"Peaks in {title_dict['measurement_name']}, {title_dict['peak_detection_method_name']}"

                ax.set_title(title_string)
                prefix = ""
                if title_dict['dataset_name']:
                    prefix = f"{title_dict['dataset_name']}_"
                figure_name = f"{prefix}overlay_{title_dict['peak_detection_method_name']}_{title_dict['measurement_name']}.png"

                if not plot_parameters.get('use_buffer', False):
                    fig.savefig(f"{figure_dir}{figure_name}", dpi=300, bbox_inches='tight', format="png" )
                return_figures.append((pdmn, title_dict['measurement_name'], save_plot_to_buffer(plot_parameters, fig), figure_name))
                # remove rects for next iteration
                for rect in added_rects: rect.remove()
        plt.close()

        return return_figures

    @staticmethod
    def _plot_overlay_best_features_aligned_classwise(mcc_ims_analysis, plot_parameters):
        """
        Plot classwise intensity_matrix and detected peaks on top with deviation indicated with seethrough rects - best features in blue
        :param mcc_ims_analysis:
        :param plot_parameters:
        :return:
        """
        print("Plotting Overlay Best Features Plots")
        collected_figure_tuples = []
        peak_alignment_result = mcc_ims_analysis.peak_alignment_result

        title_dict = {'dataset_name': mcc_ims_analysis.dataset_name}

        class_labels = np.unique([m.class_label for m in mcc_ims_analysis.measurements])
        class_label_dict = {cl: [] for cl in class_labels}
        for m in mcc_ims_analysis.measurements:
            class_label_dict[m.class_label].append(m)

        # create consensus df for each class
        consensus_dfs = {}
        for class_label, measurements in class_label_dict.items():
            consensus_dfs[class_label] = HeatmapPlot.compute_averaged_df([m.df for m in measurements])

        peak_alignment_result = mcc_ims_analysis.peak_alignment_result

        for class_label, consensus_df in consensus_dfs.items():
            # title_dict['class_label'] = class_label
            class_str = "best_by_class-{}".format(class_label.replace(" ", ""))
            print("class_str={}".format(class_str))
            print("class_label={}".format(class_label))
            title_dict['measurement_name'] = class_str
            # speed up by iterating over measurement, updating rects and keeping heatmap
            collected_figure_tuples.extend(
                (ClusterPlot._plot_overlay_clustering_helper_fast(
                    intensity_matrix=consensus_df, pdm_coord_tuples=peak_alignment_result.peak_coordinates.items(),
                    plot_parameters=plot_parameters, title_dict=title_dict, best_features_df=mcc_ims_analysis.analysis_result.best_features_df))
            )
        return collected_figure_tuples

    @staticmethod
    def _plot_overlay_best_features(mcc_ims_analysis, plot_parameters):
        """
        Plot intensity_matrix and detected peaks on top with deviation indicated with seethrough rects - best features in blue
        :param mcc_ims_analysis:
        :param plot_parameters:
        :return:
        """
        print("Plotting Overlay Best Features Plots")
        collected_figure_tuples = []
        # could compute intersection of all and best features
        # using abstract function for adding peak coords
        peak_alignment_result = mcc_ims_analysis.peak_alignment_result
        title_dict = {'dataset_name': mcc_ims_analysis.dataset_name}
        for m in mcc_ims_analysis.measurements:
            parsed_filename = "{}_ims.csv".format(m.filename.split("_ims")[0])
            title_dict['measurement_name'] = parsed_filename
            # speed up by iterating over measurement, updating rects and keeping heatmap
            collected_figure_tuples.extend(
                (ClusterPlot._plot_overlay_clustering_helper_fast(
                    intensity_matrix=m.df, pdm_coord_tuples=peak_alignment_result.peak_coordinates.items(),
                    plot_parameters=plot_parameters, title_dict=title_dict, best_features_df=mcc_ims_analysis.analysis_result.best_features_df))
            )
        return collected_figure_tuples


    @staticmethod
    def _plot_overlay_aligned_clustering(mcc_ims_analysis, plot_parameters):
        """
        Plot intensity_matrix and detected peaks on top (red rects)
        :param mcc_ims_analysis:
        :param plot_parameters:
        :return:
        """
        print("Plotting Overlay Plots")
        collected_figure_tuples = []
        peak_alignment_result = mcc_ims_analysis.peak_alignment_result
        title_dict = {'dataset_name': mcc_ims_analysis.dataset_name}
        for m in mcc_ims_analysis.measurements:

            # remove double .csv and _preprocessed in name

            preprocessed_filename = m.filename.split(".csv.csv")[0]
            preprocessed_filename = preprocessed_filename.split("_preprocessed")[0]

            title_dict['measurement_name'] = preprocessed_filename
            # speed up by iterating over measurement, updating rects and keeping heatmap
            collected_figure_tuples.extend(
                    (ClusterPlot._plot_overlay_clustering_helper_fast(
                    intensity_matrix=m.df, pdm_coord_tuples=peak_alignment_result.peak_coordinates.items(),
                    plot_parameters=plot_parameters, title_dict=title_dict))
                )
        return collected_figure_tuples


    @staticmethod
    def _plot_overlay_aligned_clustering_classwise(mcc_ims_analysis, plot_parameters):
        """
        Plot intensity_matrix and detected peaks on top of averaged measurement by class_label (red rects)
        :param mcc_ims_analysis:
        :param plot_parameters:
        :return:
        """
        print("Plotting Classwise Overlay Plots")
        collected_figure_tuples = []
        peak_alignment_result = mcc_ims_analysis.peak_alignment_result

        title_dict = {'dataset_name': mcc_ims_analysis.dataset_name}

        class_labels = np.unique([m.class_label for m in mcc_ims_analysis.measurements])
        class_label_dict = {cl: [] for cl in class_labels}
        for m in mcc_ims_analysis.measurements:
            class_label_dict[m.class_label].append(m)

        # create consensus df for each class
        consensus_dfs = {}
        for class_label, measurements in class_label_dict.items():
            consensus_dfs[class_label] = HeatmapPlot.compute_averaged_df([m.df for m in measurements])

        for class_label, consensus_df in consensus_dfs.items():
            # title_dict['class_label'] = class_label
            class_str = "by_class-{}".format(class_label.replace(" ",""))
            print("class_str={}".format(class_str))
            print("class_label={}".format(class_label))
            title_dict['measurement_name'] = class_str
            # speed up by iterating over measurement, updating rects and keeping heatmap
            collected_figure_tuples.extend(
                    (ClusterPlot._plot_overlay_clustering_helper_fast(
                    intensity_matrix=consensus_df, pdm_coord_tuples=peak_alignment_result.peak_coordinates.items(),
                    plot_parameters=plot_parameters, title_dict=title_dict))
                )
        return collected_figure_tuples

    ClusterBasic = _plot_clustering
    ClusterMultiple = _plot_multiple_peak_detection_methods_clustering
    OverlayClasswiseAlignment = _plot_overlay_aligned_clustering_classwise
    OverlayAlignment = _plot_overlay_aligned_clustering
    OverlayBestFeaturesAlignment = _plot_overlay_best_features
    OverlayBestFeaturesClasswiseAlignment = _plot_overlay_best_features_aligned_classwise


class FeatureMatrixPlot(object):
    """
    Collection of Feature matrix visualizations
    """

    @staticmethod
    def _plot_patients_vs_features(df, index_label_mapping, measurement_mapping, plot_parameters):
        """
        Plot heatmap of patient csv_names vs peaks,
        :param df: a matrix with encoded features
        :type df: pd.Dataframe
        :param index_label_mapping:
        :return:
        """
        plot_prefix = ''

        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']

        # print(m_train.head())
        # TODO make plot generic for multi label
        # save class labels independent from names of classes
        class_labels = np.unique([a for a in index_label_mapping.values()])
        number_of_class_labels = len(class_labels)

        if number_of_class_labels > 2:
            print("Plot only available for two class labels, will use first two.")
        print(class_labels)
        print(number_of_class_labels)
        map_labels = {l: [] for l in class_labels}
        for k, v in index_label_mapping.items(): map_labels[v].extend([k]);

        # plot a heatmap
        plt.close('all')
        fig, axn = plt.subplots(2, 1, sharex=True, figsize=(9, 4))
        bw_colormap = ListedColormap(['black', 'white'])
        for i, ax in enumerate(axn.flat):
            if i == 0:
                # sns.heatmap(df.loc[citrus_labels], yticklabels=citrus_yticks, xticklabels=15, cbar=False, ax=ax)
                # sns.heatmap(df.loc[map_labels[class_labels[0]]], yticklabels=[measurement_mapping[index_number] for index_number in map_labels[class_labels[0]]], xticklabels=15, cbar=False, ax=ax)
                sns.heatmap(df.loc[map_labels[class_labels[0]]],
                            yticklabels=[measurement_mapping[index_number] for index_number in map_labels[class_labels[0]]],
                            xticklabels=15, cmap=bw_colormap, ax=ax, cbar=False)
                ax.set_ylabel("citrus")
                ax.set_title('Patients vs Features (Peaks)')
            else:
                # sns.heatmap(df.loc[original_labels], yticklabels=original_yticks, xticklabels=15, cbar=False, ax=ax)
                sns.heatmap(df.loc[map_labels[class_labels[1]]],
                            yticklabels=[measurement_mapping[index_number] for index_number in map_labels[class_labels[1]]],
                            xticklabels=15, cmap=bw_colormap, cbar=False, ax=ax)
                ax.set_ylabel("original")
            plt.sca(ax)
            plt.yticks(rotation=0)

        # set title
        # fig.subtitle('Features (Peaks) vs Patients')
        # configure legend
        black_patch = matplotlib.patches.Patch(color='black', label='0')
        white_patch = matplotlib.patches.Patch(color='white', label='1')
        plt.legend(title='Peak present', handles=[black_patch, white_patch], bbox_to_anchor=(1.2, 1.3))
        # fig.tight_layout()
        plt.savefig("%sheatmaps/%s_patients_vs_peaks.png" % (plot_parameters['plot_dir'], plot_prefix), bbox_inches='tight')
        # plt.close('all')
        # # sns.heatmap(m_train[original_labels], ax=ax)
        #
        # # print(m_train)
        # # print(feature_index)
        # # print(feature_index)
        plt.close('all')


    BitMatrix = _plot_patients_vs_features


class MaskPlot(object):
    """
    Collection of mask visualizations after peak detection
    """

    @staticmethod
    def _plot_mask_from_peak_detection_result(peak_detection_result, plot_parameters, ):
        """
        Visualize peak detection result of PeakDetectionResult
        :type peak_detection_result: model.BreathCore.PeakDetectioResut
        :param peak_detection_result:
        :param plot_parameters:
        :return:
        """
        measurement_name = peak_detection_result.outname
        MaskPlot._plot_mask(peak_detection_result.peak_detection_mask, measurement_name, plot_parameters)

    @staticmethod
    def _plot_mask(intensity_matrix, measurement_name, plot_parameters):
        # TODO mention the peak detection method in a generic way - right now only available from TOPHAT
        print("Plotting mask from {} created with {}".format(measurement_name, "TopHat"))
        plot_prefix = ''
        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']

        """Set the x- and y-axis labels"""
        # xmax = intensity_matrix['inverse_reduced_mobility'].max()
        # pdb.set_trace()
        inverse_reduced_mobility_vals = intensity_matrix.index.values
        retention_time_vals = intensity_matrix.columns.values[:-1]
        xstep = [str(n) for n in np.arange(np.around(inverse_reduced_mobility_vals.min(), decimals=1),
                                           inverse_reduced_mobility_vals.max(), 0.1)]
        # ymax = intensity_matrix['retention_time'].max()

        ystep = [str(ys) for ys in np.arange(0.0, retention_time_vals.max(), 25)]
        majorsx = xstep
        majorsy = ystep
        """Create the heatmap"""
        ax = sns.heatmap(intensity_matrix.T, cmap=plot_parameters['colormap'],
                         xticklabels=len(inverse_reduced_mobility_vals) // len(majorsx),
                         yticklabels=len(retention_time_vals) // len(majorsy))
        ax.set_xlabel('1/K0')
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(majorsx))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(majorsy))
        plt.yticks(rotation=0)
        sns.set_style('ticks')
        # ax.set(xlabel="Retention Time [s]", ylabel="Inverse Reduced Mobility (1/R0) [Vs/cm^2]")
        ax.set_xlabel("Inverse Reduced Mobility (1/k0) [Vs/cm^2]")
        ax.set_ylabel("Retention Time [s]")
        ax.set_title('Measurement {}'.format(measurement_name))

        fig = ax.get_figure()
        fig.set_figsize = (9, 4)
        fig.savefig('{}masks/{}_mask_plot_{}.png'.format(plot_parameters['plot_dir'],
                                                     plot_prefix, measurement_name[:-4]),
                    bbox_inches='tight')
        fig.clf()
        plt.close('all')

    MaskHeatmap = _plot_mask_from_peak_detection_result


class RocCurvePlot(object):
    """
    Plot the ROC curve of the AnalysisResult
    """

    @staticmethod
    def _plot_two_class_roc_curve(analysis_result, plot_parameters, limit_to_peak_detection_method_name=False):
        from ..model.BreathCore import PerformanceMeasure
        return_figures = []
        print("Plotting binary ROC curve")
        # assert isinstance(analysis_result, AnalysisResult)
        plot_prefix = ''
        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']

        use_buffer = plot_parameters.get('use_buffer', False)

        figure_dir = f"{plot_parameters['plot_dir']}roc_curve"
        # first check whether directory exists for figure output - or whether we want to use a buffer
        if not use_buffer:
            Path(figure_dir).mkdir(parents=True, exist_ok=True)

        # plt.plot(analysis_result.statistics["false_positive_rates_of_splits"],
        #          analysis_result.statistics["true_positive_rates_of_splits"], lw=1, alpha=0.3, )
        # based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
        # and https://stackoverflow.com/questions/42901264/identical-auc-for-each-fold-in-cross-validation-roc-curves-python
        alignment_name = analysis_result.peak_alignment_step if analysis_result.peak_alignment_step is not None else ""
        for evaluation_method_name, statistics_by_pdm_name in analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection.items():
            # need to check first whether we computed a roc measure - only the case for RF eval
            if evaluation_method_name in [PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name]:

                for peak_detection_method_name, statistics in statistics_by_pdm_name.items():
                    if limit_to_peak_detection_method_name and limit_to_peak_detection_method_name !=peak_detection_method_name:
                        continue
                    else:
                        # clear current figure
                        fig = plt.gcf()
                        auc_measures = statistics['auc_measures']
                        # print(auc_measures.keys())

                        # need to save tprs in another way, as the splits might not be of equal sizes
                        mean_tpr = 0.0
                        mean_fpr = np.linspace(0, 1, 100)
                        # print("no of splits", len(auc_measures['tpr_of_splits_by_class'][0]))


                        # go over splits and draw fpr/tpr
                        for split_no, (tpr_per_split, fpr_per_split, auc_per_split) in enumerate(zip(
                                    auc_measures['tpr_of_splits'],
                                    auc_measures['fpr_of_splits'],
                                    auc_measures['auc_of_splits'])):

                            plt.plot(fpr_per_split, tpr_per_split, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (split_no, auc_per_split))
                            # tprs.append(interp(mean_fpr, fpr_per_split, tpr_per_split))
                            mean_tpr += interp(mean_fpr, fpr_per_split, tpr_per_split)
                            mean_tpr[0] = 0.0

                        # add random classification performance line
                        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                                 label='Random', alpha=.8)

                        # splits are not always of equal size
                        mean_tpr /= len(auc_measures['tpr_of_splits'])
                        mean_tpr[-1] = 1.0

                        mean_auc = auc(mean_fpr, mean_tpr)

                        # add the average classification line
                        plt.plot(mean_fpr, mean_tpr, color='green',
                                 label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, auc_measures['std_auc']),
                                 lw=2, alpha=.8)

                        std_tpr = np.std(mean_tpr, axis=0)
                        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                         label=r'$\pm$ 1 std. dev.')

                        # fig.add_subplot(ax)
                        RocCurvePlot._label_roc_axes(peak_detection_method_name)
                        fig = plt.gcf()

                        fig.set_figsize = (9, 4)

                        if use_buffer:
                            fig_name = f"{plot_prefix}_roc_curve_plot.png"
                        else:
                            fig_name = f"{plot_parameters['plot_dir']}roc_curve/{plot_prefix}_roc_curve_plot_{peak_detection_method_name}_{alignment_name}.png"

                        if not plot_parameters.get('use_buffer', False):
                            fig.savefig(fig_name, dpi=300, bbox_inches='tight', )

                            # ((peak_detection, model_of_class, best_feature), save_plot_to_buffer(plot_parameters, fig), fig_name))
                        return_figures.append((evaluation_method_name, peak_detection_method_name, save_plot_to_buffer(plot_parameters, fig), fig_name))
                        plt.close()
                    # fig.clf()
        return return_figures

    @staticmethod
    def _plot_multiclass_class_roc_curve(analysis_result, plot_parameters, limit_to_peak_detection_method_name=False):
        from ..model.BreathCore import PerformanceMeasure
        return_figures = []
        print("Plotting multiclass ROC curve")
        plot_prefix = ''
        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']

        use_buffer = plot_parameters.get('use_buffer', False)

        figure_dir = f"{plot_parameters['plot_dir']}roc_curve"
        # first check whether directory exists for figure output - or whether we want to use a buffer
        if not use_buffer:
            Path(figure_dir).mkdir(parents=True, exist_ok=True)


        unique_class_labels = np.sort(np.unique(analysis_result.class_labels))
        for evaluation_method_name, statistics_by_pdm_name in analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection.items():
            # need to check first whether we computed a roc measure - only the case for RF eval
            if evaluation_method_name in [PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name]:

                for peak_detection, statistics in statistics_by_pdm_name.items():
                    if limit_to_peak_detection_method_name and limit_to_peak_detection_method_name !=peak_detection:
                        continue
                    else:

                        # clear current figure
                        fig = plt.gcf()
                        auc_measures = statistics['auc_measures']
                        # dont count 'macro' key as class - ist just the average
                        number_of_classes = len(unique_class_labels)

                        # add random classification performance line
                        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

                        for class_label in range(number_of_classes):

                            # need to save tprs in another way, as the splits might not be of equal sizes
                            mean_tpr = 0.0
                            mean_fpr = np.linspace(0, 1, 100)

                            for i, (tpr_per_split, fpr_per_split, auc_per_split) in enumerate(
                                    zip(auc_measures['tpr_of_splits_by_class'][class_label],
                                        auc_measures['fpr_of_splits_by_class'][class_label],
                                        auc_measures['auc_of_splits_by_class'][class_label])):

                                # plt.plot(fpr_per_split, tpr_per_split, lw=1, alpha=0.3,
                                #          label='ROC fold %d (AUC = %0.2f)' % (i, auc_per_split))
                                # tprs.append(interp(mean_fpr, fpr_per_split, tpr_per_split))
                                mean_tpr += interp(mean_fpr, fpr_per_split, tpr_per_split)
                                mean_tpr[0] = 0.0
                                # print("split = {}".format(i))

                            # average over splits
                            mean_tpr /= len(auc_measures['tpr_of_splits_by_class'][0])
                            mean_tpr[-1] = 1.0

                            mean_auc = auc(mean_fpr, mean_tpr)

                            # add the average classification line
                            plt.plot(mean_fpr, mean_tpr,
                                     label=r'Class {} macro ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(
                                         unique_class_labels[class_label], mean_auc, auc_measures['std_auc']),
                                         # unique_class_labels[class_label], auc_measures['mean_auc'], auc_measures['std_auc']),
                                     lw=2, alpha=.8)

                            # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            #                  label=r'$\pm$ 1 std. dev.')
                        RocCurvePlot._label_roc_axes(peak_detection)
                        fig = plt.gcf()

                        pas_n = ""
                        if analysis_result.peak_alignment_step is not None:
                            pas_n = analysis_result.peak_alignment_step.name

                        fig_name = f"{plot_parameters['plot_dir']}roc_curve/{plot_prefix}_multi_roc_curve_plot_{peak_detection}_{pas_n}.png"
                        # print(fig_name)
                        if not plot_parameters.get('use_buffer', False):
                            fig.savefig(fig_name, dpi=300, bbox_inches='tight')

                        return_figures.append((evaluation_method_name, peak_detection, save_plot_to_buffer(plot_parameters, fig), fig_name))
                            # ((peak_detection, model_of_class, best_feature), save_plot_to_buffer(plot_parameters, fig), fig_name))
                        plt.close()
        return return_figures


    @staticmethod
    def _label_roc_axes(peak_detection):
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for {}'.format(peak_detection))
        plt.legend(loc="lower right")

    ROCCurve = _plot_two_class_roc_curve
    MultiClassROCCurve = _plot_multiclass_class_roc_curve

class VennDiagram(object):
    """
    Plot Venn diagrams.

    The venn import allows to create venn diagrams with more than 3 sets
    """

    @staticmethod
    def _plot_venn_diagram_from_peak_detection_result(comparison_peak_detection_result, plot_parameters):
        """
        Visualize the overlap of different PeakDetectionMethods
        :param peak_detection_result:
        :param plot_parameters:
        :return:
        """
        VennDiagram._plot_venn2_diagram(comparison_peak_detection_result, plot_parameters)
        VennDiagram._plot_venn3_diagram(comparison_peak_detection_result, plot_parameters)
        VennDiagram._plot_venn4_diagram(comparison_peak_detection_result, plot_parameters)

    @staticmethod
    def _plot_venn2_diagram(comparison_peak_detection_result, plot_parameters):
        if comparison_peak_detection_result.shape[0] > 2:
            raise AttributeError("Venn2Diagram requires 2 sets. You gave {} sets."
                                 "Choose Venn3- or Venn4Diagram, respectively".format(
                comparison_peak_detection_result.shape[0]))
        plot_prefix = ''
        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']
        peak_detections = sorted(comparison_peak_detection_result.index.values)
        individual_1 = comparison_peak_detection_result.loc[peak_detections[0]]['individual']
        individual_2 = comparison_peak_detection_result.loc[peak_detections[1]]['individual']
        intersection_12 = max([comparison_peak_detection_result.loc[peak_detections[0]][peak_detections[1]],
                               comparison_peak_detection_result.loc[peak_detections[1]][peak_detections[0]]])
        fig = plt.figure()
        v = venn2(subsets=(individual_1, individual_2, intersection_12), set_labels=peak_detections)
        c = venn2_circles(subsets=(individual_1, individual_2, intersection_12), linestyle='dashed', linewidth=1, color="grey")
        comp_str = f"venn2_{comparison_peak_detection_result.index.values[0]}_{comparison_peak_detection_result.index.values[1]}"
        fig.savefig(f"{plot_parameters['plot_dir']}venn_diagram/{plot_prefix}_{comp_str}.png", bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_venn3_diagram(comparison_peak_detection_result, plot_parameters):

        if comparison_peak_detection_result.shape[0] < 3:
            raise AttributeError("Venn3Diagram requires 3 sets. You gave {} sets."
                                 "Choose Venn2- or Venn3Diagram, respectively".format(comparison_peak_detection_result.shape[0]))
        if comparison_peak_detection_result.shape[0] > 3:
            raise AttributeError("Venn3Diagram requires 3 sets. You gave {} sets."
                                 "Choose Venn2- or Venn4Diagram, respectively".format(comparison_peak_detection_result.shape[0]))
        plot_prefix = ''
        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']
        peak_detections = sorted(comparison_peak_detection_result.index.values)
        individual_1 = comparison_peak_detection_result.loc[peak_detections[0]]['individual']
        individual_2 = comparison_peak_detection_result.loc[peak_detections[1]]['individual']
        individual_3 = comparison_peak_detection_result.loc[peak_detections[2]]['individual']
        intersection_12 = max([comparison_peak_detection_result.loc[peak_detections[0]][peak_detections[1]],
                               comparison_peak_detection_result.loc[peak_detections[1]][peak_detections[0]]])
        intersection_13 = max([comparison_peak_detection_result.loc[peak_detections[0]][peak_detections[2]],
                               comparison_peak_detection_result.loc[peak_detections[2]][peak_detections[0]]])
        intersection_23 = max([comparison_peak_detection_result.loc[peak_detections[1]][peak_detections[2]],
                               comparison_peak_detection_result.loc[peak_detections[2]][peak_detections[1]]])
        intersection_all = comparison_peak_detection_result.loc[:]['all'].max()
        fig = plt.figure()
        v = venn3(subsets=(individual_1, individual_2, intersection_12, individual_3, intersection_13, intersection_23,
                           intersection_all), set_labels=peak_detections)
        c = venn3_circles(subsets=(individual_1, individual_2, intersection_12, individual_3, intersection_13,
                                   intersection_23, intersection_all), linestyle='dashed', linewidth=1, color="grey")
        comp_str = f"venn3_{comparison_peak_detection_result.index.values[0]}_{comparison_peak_detection_result.index.values[1]}_{comparison_peak_detection_result.index.values[2]}"
        fig.savefig(f"{plot_parameters['plot_dir']}venn_diagram/{plot_prefix}_{comp_str}.png",bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_venn4_diagram(comparison_peak_detection_result, plot_parameters):
        from breathpy.view import venn

        if comparison_peak_detection_result.shape[0] < 4:
            raise AttributeError("Venn4Diagram requires 4 sets. You gave {} sets."
                                 "Choose Venn2- or Venn3Diagram, respectively".format(comparison_peak_detection_result.shape[0]))
        if comparison_peak_detection_result.shape[0] > 4:
            raise AttributeError("Venn4Diagram requires 4 sets. You gave {} sets."
                                 "A VennDiagram with more than 4 sets is to complex. Choose another from of visualization".format(comparison_peak_detection_result.shape[0]))
        plot_prefix = ''
        if 'plot_prefix' in plot_parameters:
            plot_prefix = plot_parameters['plot_prefix']
        labels = dict()
        peak_detections = sorted(comparison_peak_detection_result.index.values)
        labels['1000'] = "{}".format(comparison_peak_detection_result.loc[peak_detections[0]]['individual'])
        labels['0100'] = "{}".format(comparison_peak_detection_result.loc[peak_detections[1]]['individual'])
        labels['0010'] = "{}".format(comparison_peak_detection_result.loc[peak_detections[2]]['individual'])
        labels['0001'] = "{}".format(comparison_peak_detection_result.loc[peak_detections[3]]['individual'])
        labels['1100'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[0]][peak_detections[1]],
                                    comparison_peak_detection_result.loc[peak_detections[1]][peak_detections[0]]]))
        labels['1010'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[0]][peak_detections[2]],
                                    comparison_peak_detection_result.loc[peak_detections[2]][peak_detections[0]]]))
        labels['1001'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[0]][peak_detections[3]],
                                    comparison_peak_detection_result.loc[peak_detections[3]][peak_detections[0]]]))
        labels['0110'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[1]][peak_detections[2]],
                                    comparison_peak_detection_result.loc[peak_detections[2]][peak_detections[1]]]))
        labels['0101'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[1]][peak_detections[3]],
                                    comparison_peak_detection_result.loc[peak_detections[3]][peak_detections[1]]]))
        labels['0011'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[2]][peak_detections[3]],
                                    comparison_peak_detection_result.loc[peak_detections[3]][peak_detections[2]]]))
        labels['1110'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[0]]["{}_{}".format(peak_detections[1], peak_detections[2])],
                                    comparison_peak_detection_result.loc[peak_detections[1]]["{}_{}".format(peak_detections[0], peak_detections[2])],
                                    comparison_peak_detection_result.loc[peak_detections[2]]["{}_{}".format(peak_detections[0], peak_detections[1])]]))
        labels['1101'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[0]]["{}_{}".format(peak_detections[1], peak_detections[3])],
                                    comparison_peak_detection_result.loc[peak_detections[1]]["{}_{}".format(peak_detections[0], peak_detections[3])],
                                    comparison_peak_detection_result.loc[peak_detections[3]]["{}_{}".format(peak_detections[0], peak_detections[1])]]))
        labels['1011'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[0]]["{}_{}".format(peak_detections[2], peak_detections[3])],
                                    comparison_peak_detection_result.loc[peak_detections[2]]["{}_{}".format(peak_detections[0], peak_detections[3])],
                                    comparison_peak_detection_result.loc[peak_detections[3]]["{}_{}".format(peak_detections[0], peak_detections[2])]]))
        labels['0111'] = "{}".format(max([comparison_peak_detection_result.loc[peak_detections[1]]["{}_{}".format(peak_detections[2],peak_detections[3])],
                                    comparison_peak_detection_result.loc[peak_detections[2]]["{}_{}".format(peak_detections[1], peak_detections[3])],
                                    comparison_peak_detection_result.loc[peak_detections[3]]["{}_{}".format(peak_detections[1], peak_detections[2])]]))
        labels['1111'] = "{}".format(comparison_peak_detection_result.loc[:]['all'].max())
        fig, ax = venn.venn4(labels=labels, names=peak_detections)
        fig.savefig(f"{plot_parameters['plot_dir']}venn_diagram/{plot_prefix}_venn4.png", bbox_inches='tight')
        plt.close()

    Venn2Diagram = _plot_venn2_diagram
    Venn3Diagram = _plot_venn3_diagram
    Venn4Diagram = _plot_venn4_diagram


class BoxPlot(object):
    """
    Create Box plots
    """

    @staticmethod
    def convert_pval(n):
        return '%.1E' % Decimal(n)

    @staticmethod
    def _plot_boxplot_of_best_features_seaborn(analysis_result, plot_parameters, limit_to_peak_detection_method_name=False):
        from ..model.BreathCore import PerformanceMeasure
        from ..model.BreathCore import MccImsAnalysis
        import statannot
        sns.set_style("whitegrid")
        # pdb.set_trace()
        # types dont match when using in web, as we use different imports
        # assert isinstance(analysis_result, AnalysisResult)
        plots_to_return = []
        print("Plotting Boxplots of the best features SEABORN")
        figure_dir = f"{plot_parameters['plot_dir']}boxplots"
        # first check whether directory exists for figure output - or whether we want to use a buffer
        if not plot_parameters.get('use_buffer', False):
            Path(figure_dir).mkdir(parents=True, exist_ok=True)

        corrected_pvalues_df = analysis_result.get_pvalues_df(
            n_features=MccImsAnalysis.DEFAULT_PERFORMANCE_MEASURE_PARAMETERS[PerformanceMeasure.FDR_CORRECTED_P_VALUE][
                'n_of_features'])


        for peak_detection_name, peak_intensities_by_class_by_peak in analysis_result.peak_intensities_by_pdm_by_class_by_peak.items():
            if limit_to_peak_detection_method_name and peak_detection_name != peak_detection_name:
                plt.close()
                continue
            else:
                corrected_pvals_by_pdmn = corrected_pvalues_df[
                        corrected_pvalues_df['peak_detection_method_name'] == peak_detection_name]
                #
                # classes = peak_intensities_by_class_by_peak.keys()
                #
                # # classes have the same peak_ids
                # peak_ids = sorted(peak_intensities_by_class_by_peak[classes[0]].keys())
                best_feature_set = set(np.unique(corrected_pvals_by_pdmn['peak_id']))
                records = []
                for model_of_class, intensities_by_pid in peak_intensities_by_class_by_peak.items():
                    for peak_id, intensities in intensities_by_pid.items():

                        # only create record if peak_id in best_features - otherwise we get too many plots and it takes forever
                        if peak_id in best_feature_set:
                            record_dict = {}
                            record_dict['class_str'] = model_of_class
                            record_dict['peak_id'] = peak_id
                            record_dict['intensities'] = intensities
                            records.append(record_dict)
                df = pd.DataFrame.from_records(records)

                # classes = np.unique(df['class_str'].values)
                # peak_ids = np.unique(df['peak_id'].values)

                for peak_id, rows in df.groupby('peak_id'):
                    # pdb.set_trace()
                    list_of_intensities_of_classes = [*rows['intensities'].values]
                    class_strs = [*rows['class_str'].values]

                    #  make seaborn boxplot here
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    # need to unpack intensities for seaborn
                    unpacked_df_lis = []
                    for class_str, class_rows in rows.groupby('class_str'):
                        data = {"Intensity": class_rows.intensities.values[0],
                                "class": [class_str]*len(class_rows.intensities.values[0])}
                        unpacked_df_lis.append(pd.DataFrame(data))
                    unpacked_df = pd.concat(unpacked_df_lis)

                    # ax = sns.boxplot(data=unpacked_df, x='class', y="Intensity", hue="class", ax=ax)
                    # sort so colors match between decision tree and boxplots
                    ax = sns.boxplot(data=unpacked_df, x='class', y="Intensity", hue="class", ax=ax, hue_order=np.unique(unpacked_df['class']))

                    pval_rows = corrected_pvals_by_pdmn[corrected_pvals_by_pdmn['peak_id'] == peak_id]
                    cor_pvals = []
                    reg_pvals = []
                    box_pairs = []
                    class_comparisons = pval_rows['class_comparison']
                    y = 0.035

                    for class_comparison in class_comparisons:
                        pval_row = pval_rows[pval_rows['class_comparison'] == class_comparison]

                        p_value = pval_row['raw_p_values'].values[0]
                        corrected_pval = pval_row['corrected_p_values'].values[0]

                        if len(class_comparisons) == 1:
                            reg_pvals.append(p_value)
                            cor_pvals.append(corrected_pval)
                            classes = class_comparison.split("_VS_")
                            box_pairs.append(((classes[0],classes[0]), (classes[1],classes[1])))
                        else:
                            # convert pvalue to scientific notation instead of <0.001
                            if p_value < 0.001:
                                plt.figtext(x=0.03, y=y,
                                            s=f"{class_comparison}: p = {BoxPlot.convert_pval(p_value)} | fdr_corrected = {BoxPlot.convert_pval(corrected_pval)}")
                            else:
                                plt.figtext(x=0.03, y=y, s="{}: p ={:.4f}  | fdr_corrected = {:.4f}".format(class_comparison, p_value, corrected_pval))
                            y -= 0.04

                    if len(class_comparisons) == 1:
                        # only works in binary case - one vs rest doesnt make sense in this annotation
                        statannot.add_stat_annotation(
                            ax, data=unpacked_df, x='class', y='Intensity', hue='class',
                            box_pairs=box_pairs, text_format='full',
                            perform_stat_test=False, pvalues=cor_pvals, loc='inside', verbose=0)

                        text_annot_custom = "P-Values are FDR corrected using the Benjamini-Hochberg method"
                        plt.figtext(x=0.03, y=y, s=text_annot_custom)

                    # remove x-axis label
                    ax.set_xlabel('')
                    # put legend outside plot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    plt.title(peak_id)

                    if plot_parameters.get('use_buffer', False):
                        fig_name = f"{Path(plot_parameters['plot_prefix']).stem}_bp_{peak_id}.png"
                    else:
                        fig_name = f"{figure_dir}/{plot_parameters['plot_prefix']}_{peak_detection_name}_model_of_class_{model_of_class}_{peak_id}.png"

                    fig = plt.gcf()
                    plots_to_return.append(((PerformanceMeasure.FDR_CORRECTED_P_VALUE.name, peak_detection_name,
                                             model_of_class, peak_id), save_plot_to_buffer(plot_parameters, fig),
                                            fig_name))
                    if not plot_parameters.get('use_buffer', False):
                        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
                    plt.close()

        return plots_to_return

    BoxPlotBestFeature = _plot_boxplot_of_best_features_seaborn


class TreePlot(object):
    """
    Create plot of tree with graphviz
    """
    @staticmethod
    def _plot_decision_trees_from_analysis_result(analysis_result, plot_parameters, limit_to_peak_detection_method_name=False):
        print(f"Plotting Decision trees buffer = {plot_parameters.get('use_buffer', False)} for {analysis_result}")
        # plot decision tree for each performance_measure and peak detection method
        # assert isinstance(analysis_result, AnalysisResult)
        if plot_parameters.get('use_buffer', False):
            tmp = os.path.join(tempfile.gettempdir(), '.breath/{}'.format(hash(os.times())))
            os.makedirs(tmp)
            plot_parameters['plot_dir'] = tmp

        # create temporary dir to save plots, read them back in, delete tempdir
        plots = []
        dot_data_full = analysis_result.decision_tree_per_evaluation_method_and_peak_detection
        for eval_method_name, by_pdm_dot_data in dot_data_full.items():
            for pdm_name, dot_data_lis in by_pdm_dot_data.items():

                # forgot the else case for non-automatic workflow - limit is not set
                if (limit_to_peak_detection_method_name and (limit_to_peak_detection_method_name == pdm_name)) or (not limit_to_peak_detection_method_name):
                    plots.append(TreePlot._plot_decision_tree_helper(dot_data_lis, eval_method_name=eval_method_name, pdm_name=pdm_name, plot_parameters=plot_parameters))

        if plot_parameters.get('use_buffer', False):
            shutil.rmtree(tmp, ignore_errors=True)
        return plots

    @staticmethod
    def _plot_decision_tree_helper(dot_data, eval_method_name, pdm_name, plot_parameters):
        figure_dir = f"{plot_parameters['plot_dir']}/decision_trees"
        fig_path = f"{figure_dir}/{plot_parameters['plot_prefix']}_{eval_method_name}_{pdm_name}"
        rv = []
        for class_comparison_str, dots in dot_data:
            if plot_parameters.get('use_buffer', False):
                # fig_name = f"{Path(plot_parameters['plot_prefix']).stem}_dt_{class_comparison_str}.png"
                # fig_name = f"{Path(plot_parameters['plot_prefix']).stem}_dt.png"
                # fig_name = f"{Path(plot_parameters['plot_prefix']).stem}_dt.png"
                fig_name = f"{Path(fig_path).stem}_dt.png"
            else:
                # fig_name = f"{fig_path}_{class_comparison_str}.png"
                fig_name = f"{fig_path}.png"
            rv.append(((eval_method_name, pdm_name, class_comparison_str), TreePlot._render_decision_tree_graph(dots, fig_name, plot_parameters), fig_name))
        return rv

    @staticmethod
    def _render_decision_tree_graph(dot_data, path, plot_parameters):
        rv = None
        graph = graphviz.Source(source=dot_data, format="png")
        graph.render(path[:-4], cleanup=True)

        # if use buffer - we write to temp dir and have to read them back in
        if plot_parameters.get('use_buffer', False):
            with open(path, 'rb') as bytes_handle:
                rv = BytesIO(bytes_handle.read())
        return rv

    DecisionTrees = _plot_decision_trees_from_analysis_result





def save_plot_to_buffer(plot_parameters, fig, dpi=200):
    rv = None
    if plot_parameters.get('use_buffer', False):
        buffer = BytesIO()
        fig.savefig(buffer, bbox_inches='tight', dpi=dpi)
        rv = buffer
    return rv
