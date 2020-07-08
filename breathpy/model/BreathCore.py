
"""
Module defining the core elements of breathpy
"""
__author__ = 'philmaweb, aaron_ruben'

from abc import ABCMeta, abstractmethod
import warnings
import functools
import multiprocessing
import subprocess
from collections import OrderedDict, defaultdict
import decimal
import glob
from functools import wraps
from pathlib import Path
from itertools import compress, combinations, chain
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import white_tophat
from tempfile import NamedTemporaryFile
from io import StringIO, BytesIO

import tempfile
import os
from shutil import rmtree, copyfile

import pywt
import zipfile
from scipy import signal
from sklearn.cluster import (KMeans,
                             DBSCAN,
                             AffinityPropagation,
                             MeanShift)
from sklearn.model_selection import cross_validate


from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from sklearn import tree, metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests

from collections import Counter
from matplotlib import colors

from .ProcessingMethods import (
    DenoisingMethod, NormalizationMethod,
    PeakDetectionMethod, ExternalPeakDetectionMethod, PeakAlignmentMethod,
    PerformanceMeasure, FeatureReductionMethod,
    GCMSPeakDetectionMethod, GCMSAlignmentMethod,)

from .GCMSTools import (
        run_centroid_sample, run_raw_wavelet_sample, align_feature_xmls, convert_consensus_map_to_feature_matrix,
        filter_mzml_or_mzxml_filenames)


def handle_peak_alignment_result_type(peak_alignment_func):
    """
    Create either `FloatPeakAlignmentResult` or `BitPeakAlignmentResult` from peak_alignment function
    :param peak_alignment_func:
    :return:
    """
    def _func_wrapper(*args, **kwargs):
        if "alignment_result_type" in kwargs:
            alignment_result_type = kwargs.pop('alignment_result_type')

            if 'peak_alignment_step' in kwargs:
                peak_alignment_step = kwargs.pop('peak_alignment_step')
            else:
                peak_alignment_step = PeakAlignmentMethod.PROBE_CLUSTERING

            if issubclass(alignment_result_type, BitPeakAlignmentResult):
                print("Handling BitPeakAlignmentResult for {}".format(peak_alignment_func.__name__))
                return BitPeakAlignmentResult(*peak_alignment_func(*args, **kwargs), **kwargs)
            elif issubclass(alignment_result_type, FloatPeakAlignmentResult):
                print("Handling FloatPeakAlignmentResult for {}".format(peak_alignment_func.__name__))
                # needed to remove args arguments from basic_clustering methods
                rv = peak_alignment_func(*args, **kwargs)
                return FloatPeakAlignmentResult(*rv, peak_alignment_step=peak_alignment_step, **kwargs)
            else:
                # if alignment_result_type == type(FloatPeakAlignmentResult):
                raise NotImplementedError("Class {} not supported! Use {} or {} instead.".format(
                    alignment_result_type, BitPeakAlignmentResult, FloatPeakAlignmentResult))
        else:
            return peak_alignment_func(*args, **kwargs)
    return wraps(peak_alignment_func)(_func_wrapper)

def get_breath_analysis_dir():
    """
    Returns absolute path of main project directory - one dir above breathpy/model
    :return:
    """
    this_file_absolute_path = os.path.realpath(__file__)
    return Path(this_file_absolute_path).parent.parent

def construct_default_parameters(file_prefix, folder_name, make_plots=False, execution_dir_level='project_dir'):
    """
    Construct default plot and file parameters. Either execute from project home directory or breathpy/model.
        Use `execution_dir_level`='one' if executing from project home - otherwise leave as is.
    :param file_prefix: what to prefix created plots and exports - eg "train_full_candy"
    :param folder_name: where data is located - eg "train_full_candy"
    :param make_plots: Are plots created - can be time consuming
    :param execution_dir_level: Will influence where to look for data and where plots are saved
    :return:
    """
    dir_level = ""
    if execution_dir_level == "one":  # where lies the execution of the python script? do wee need to add ../ to our results and data directory?
        dir_level = "../"

    colormap = colors.LinearSegmentedColormap.from_list('mcc/ims', ['white', 'blue', 'red', 'yellow'])

    plot_parameters = {'make_plots': make_plots, 'plot_prefix': file_prefix,
                       'plot_dir': '{}results/plots/'.format(dir_level),
                       'colormap': colormap,
                       }
    results_dir = '{}results/'.format(dir_level)
    data_dir = '{}data/'.format(dir_level)
    folder_path = data_dir + folder_name + "/"
    label_filename = '{}data/{}/class_labels.csv'.format(dir_level, folder_name)

    project_dir = str(Path(get_breath_analysis_dir())) + "/"
    file_parameters = {
                       'dir_level': dir_level,
                       'project_dir': project_dir,
                       'label_filename': label_filename,
                       'visualnow_layer_filename': '{}data/{}/{}_layer.xls'.format(dir_level, folder_name, file_prefix),
                       'folder_path': folder_path,
                       'folder_name': folder_name,
                       'data_dir': data_dir,
                       'results_dir': results_dir,
                       'pickle_dir': '{}{}'.format(results_dir, "pickles/"),
                       'plot_dir': '{}results/plots/'.format(dir_level),
                       'file_prefix': file_prefix,
                       'out_dir': '{}{}{}'.format(results_dir, "data/", "{}/".format(folder_name)),
                       }
    return plot_parameters, file_parameters


def construct_custom_processing_evaluation_dict(min_num_cv=2):
    """
    Return custom preprocessing and evaluation options
    :param min_num_cv:
    :return:
    """
    rf = MccImsAnalysis.DEFAULT_PERFORMANCE_MEASURE_PARAMETERS[PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION]
    rf['n_splits_cross_validation'] = min_num_cv
    return {ExternalPeakDetectionMethod.CUSTOM.name : {},
             PeakAlignmentMethod.PROBE_CLUSTERING.name : {},
             },{

            PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name: rf,
            PerformanceMeasure.FDR_CORRECTED_P_VALUE.name: MccImsAnalysis.DEFAULT_PERFORMANCE_MEASURE_PARAMETERS[PerformanceMeasure.FDR_CORRECTED_P_VALUE],
            FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES.name: MccImsAnalysis.DEFAULT_EVALUATION_MEASURE_PARAMETERS[FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES]
        }


def construct_default_processing_evaluation_steps(min_num_cv=3):
    """
    Create the default preprocessing and evaluation steps as defined in `MccImsAnalysis`
    :return:
    """
    rf_params = MccImsAnalysis.DEFAULT_PERFORMANCE_MEASURE_PARAMETERS[PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION]
    fdr_params = MccImsAnalysis.DEFAULT_PERFORMANCE_MEASURE_PARAMETERS[PerformanceMeasure.FDR_CORRECTED_P_VALUE]
    if min_num_cv != 3:
        rf_params['n_splits_cross_validation'] = min_num_cv
        fdr_params['n_splits_cross_validation'] = min_num_cv

    return [
        NormalizationMethod.INTENSITY_NORMALIZATION,
        NormalizationMethod.BASELINE_CORRECTION,
        DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY,
        DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION,
        DenoisingMethod.SAVITZKY_GOLAY_FILTER,
        DenoisingMethod.MEDIAN_FILTER,
        DenoisingMethod.GAUSSIAN_FILTER,
        PeakDetectionMethod.WATERSHED,
        # PeakDetectionMethod.JIBB,
        ExternalPeakDetectionMethod.PEAX,
        PeakDetectionMethod.VISUALNOWLAYER,
        PeakDetectionMethod.TOPHAT,
        PeakAlignmentMethod.PROBE_CLUSTERING,
        # PeakAlignmentMethod.DB_SCAN_CLUSTERING,
    ], {
        PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION: rf_params,
        PerformanceMeasure.FDR_CORRECTED_P_VALUE: fdr_params,
        PerformanceMeasure.DECISION_TREE_TRAINING: MccImsAnalysis.DEFAULT_PERFORMANCE_MEASURE_PARAMETERS[PerformanceMeasure.DECISION_TREE_TRAINING],
        FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES: MccImsAnalysis.DEFAULT_FEATURE_REDUCTION_PARAMETERS[FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES],
    }


class PeakDetectionError(ValueError):
    """
    Error when applying Peak Detection
    """

class ResumeAnalysisError(ValueError):
    """
    Error when importing Peak Detection results
    """

class ExportError(ValueError):
    """
    Error when exporting preliminary results
    """

class InvalidLayerFileError(ValueError):
    """
    Error when parsing an invalid peak layer
    """

class Measurement(object):
    """ A measurement of a breath sample
    """

    __metaclass__ = ABCMeta

    def __init__(self, df, header, is_raw=True, class_label=None):
        """
        :type header: dict
        :param header: Parsed header of Measurement file
        :type is_raw: bool
        :param is_raw: Flag to
        :type df: pd.DataFrame
        :param df: dataframe
        """
        self.df = df
        self.header = header
        self.is_raw = is_raw
        self.class_label = class_label


    @abstractmethod
    def parse_raw_measurement(self, filename):
        raise NotImplementedError()


    @abstractmethod
    def align_scale_measurements(self, alignment_scaling_methods):
        raise NotImplementedError()


    @abstractmethod
    def normalize_measurement(self, normalization_methods):
        raise NotImplementedError()


    @abstractmethod
    def denoise_measurement(self, denoising_methods):
        raise NotImplementedError()


    @abstractmethod
    def get_description(self):
        raise NotImplementedError()


    def set_class_label(self, label):
        """
        Set field of measurement to label - but make sure the class labels are ordered alphabetically
        :return:
        """
        self.class_label = label


    def get_df(self):
        return self.df


class MccImsMeasurement(Measurement):
    """
    A MccImsMeasurement
    """

    # measurement_names = []

    def __init__(self, raw_filename, class_label=None, init_df=None):
        self.raw_filename = raw_filename
        # if we pass a file from memory or from a bufer, we shouldn't fail
        # parsing from path will create a str
        # parsing from zip file will create BytesIO or StringIO
        # parsing from file will create an object with readlines
        if (isinstance(raw_filename, (str, BytesIO, StringIO)) or hasattr(raw_filename, "readlines")) and (init_df is None):
            print(f"Parsing Measurement {raw_filename}")
            # define what axis represents what
            csv_dic = self.parse_raw_ims_measurement(raw_filename)
            inverse_reduced_mobility = csv_dic['inverse_reduced_mobility']
            header = csv_dic['header']
            retention_times = csv_dic['retention_times']
            intensities = np.vstack(csv_dic['intensities'])
            intensity_matrix = pd.DataFrame(columns=retention_times, index=inverse_reduced_mobility, data=intensities)
            tDcorr = csv_dic['tDcorr']
            is_raw = True
        elif (init_df is not None) and isinstance(init_df, pd.DataFrame):
            intensity_matrix = init_df
            filename = str(self.raw_filename.rsplit("_preprocessed.csv")[0] + ".csv").rsplit("/", maxsplit=1)[-1]
            retention_times = intensity_matrix.columns.values
            inverse_reduced_mobility = intensity_matrix.index.values
            header = {'filename': filename}
            is_raw = False
            tDcorr = np.array([])
        else:
            raise ResumeAnalysisError(f"Can't initialize MccImsMeasurement. pandas.DataFrame required, got {type(raw_filename)} instead.")
            # intensity_matrix = pd.DataFrame(columns=inverse_reduced_mobility, index=retention_times, data=intensities)
        # df = pd.DataFrame(np.vectorize(np.abs)(df))
        super().__init__(intensity_matrix, header, is_raw=is_raw, class_label=class_label)
        self.inverse_reduced_mobility = inverse_reduced_mobility
        self.retention_times = retention_times
        self.tDcorr = tDcorr
        self.max_peak_intensity = 0.0
        self.filename = header.get('filename', str(raw_filename).rsplit("/", maxsplit=1)[-1])
        # add ims filename to measurement_names
        # MccImsMeasurement.measurement_names.append(self.filename)
        # File processing parameters


    def __str__(self):
        if self.class_label is not None:
            return "MccImsMeasurement {0} : {1}".format(self.header['filename'], self.class_label)
        return "MccImsMeasurement {0}".format(self.header['filename'])


    def __repr__(self):
        return str(self)


    def normalize_measurement(self, normalization_methods):
        # map NormalizationMethods to functions
        normalization_map = {
            NormalizationMethod.INTENSITY_NORMALIZATION: self.normalize_by_intensity,
            NormalizationMethod.BASELINE_CORRECTION: self.baseline_correction_rip_compensation,
        }
        for nm in normalization_methods:
            print("Applying Normalization {}".format(nm))
            normalization_map[nm]()


    def denoise_measurement(self, denoising_methods, param_map={}):
        denoise_map = {
            DenoisingMethod.GAUSSIAN_FILTER: self.denoise_gaussian_filter,
            DenoisingMethod.NOISE_SUBTRACTION: self.denoise_noise_subtraction,
            DenoisingMethod.MEDIAN_FILTER: self.denoise_median_filter,
            DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY: self.denoise_crop_inverse_reduced_mobility,
            DenoisingMethod.SAVITZKY_GOLAY_FILTER: self.denoise_savitzky_golay_filter,
            DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION: self.denoise_discrete_wavelet_transform,
        }
        for dm in denoising_methods:
            # median filter is the slowest of the denoising methods
            print(f"Applying denoising Method {dm}")
            denoise_map[dm](**param_map.get(dm, {}))


    @staticmethod
    def create_ims_header_line_attribute_map():
        """
        Create dictionary for header of mcc ims measurement according to specifications
        :return:
        """
        return {0: "data_type",
                1: "version",
                2: "template_version",
                3: "ad_board_type",
                4: "serial_number",  # serial number of IMS device
                6: "date",  # date, DD/MM/YY
                7: "time",  # start time of measurement, hh/mm/ss
                8: "filename",  # <device serial_number>_YYMMDDhhmm_ims.csv
                11: "SAMPLE INFORMATION",
                13: "sample_type",  # sample / blank / reference
                14: "sample_id",  # string identifier, non-unique
                15: "comment",  # string --> use full field instead of only split
                16: "location",  # string
                17: "location_name",  # string
                18: "height_asl",  # meters over sea level?
                19: "total_data_acquisition_time",  #
                22: "IMS - INFORMATION",  #
                24: "operator",  #
                25: "operator_name",  #
                26: "ims",  #
                28: "k0_rip_positive",  # cm^2/Vs #position of rip
                29: "k0_rip_negative",  # cm^2/Vs
                30: "polarity",  # positive or negative
                31: "grid_opening_time",  # us, microseconds
                33: "pause",  # seconds
                34: "tD_interval_corr_start",  # drifttime correction start
                35: "tD_interval_corr_end",  # drifttime correction stop
                36: "1/k0_interval_start",  # Vs/cm^2
                37: "1/k0_interval_end",  # Vs/cm^2
                38: "number_of_data_points_per_spectra",  #
                39: "number_of_spectra",  #
                40: "number_of_averaged_spectra",  #
                41: "baseline_signal_units",  #
                42: "baseline_voltage",  #
                43: "baseline_units_per_voltage",  # = 42/41
                45: "drift_length",  # in milimeters
                46: "HV",  # something in kV
                47: "amplification",  # in V/nA
                49: "drift_gas",  # usually synth. air to make comparable between machines
                50: "drift_gas_flow",  # mililiters per minute, int
                51: "sample_gas",  # usually synth. air to make comparable between machines
                52: "sample_gas_flow",  # mililiters per minute, int
                53: "carrier_gas",  # usually synth. air to make comparable between machines
                54: "carrier_gas_flow",  # mililiters per minute, int
                55: "pre_separation_type",  # string eg MCC OV5
                56: "pre_separation_temperature",  # float in degree celsius
                57: "sample_loop_temperature",  # float in degree celsius
                58: "sample_loop_volume",  # float in mililiters
                # Ambient parameters
                60: "ambient_t_source",  #
                61: "ambient_t_degree_c",  #
                67: "ambient_p_source",  #
                68: "ambient_p_hpa",  #
                75: "6_way_valve_setting",  # eg auto
                78: "EXTERNAL SAMPLING CONTROL",
                # EXTERNAL SAMPLING CONTROL
                81: "control_status",  # splittable by |
                82: "control_zero_signal_units",  #
                83: "control_zero_voltage",  #
                84: "control_threshold_signal_units",  #
                85: "control_threshold_voltage",  #
                86: "control_threshold2_signal_units",  #
                87: "control_threshold2_voltage",  #
                88: "control_sampling_time_seconds",  #
                89: "control_variable",  # eg Flow
                90: "control_dimension",  # a.u.
                # STATISTICS
                97: "STATISTICS",
                99: "rip_detection",  # enabled or disabled - flags whether to pass following data
                100: "td_rip_correction_milliseconds",  #
                101: "1/k0_rip",  #
                102: "k0_rip",  #
                103: "snr_rip",  #
                104: "whm_rip",  #
                105: "res_power_rip",  #
                107: "td_pre_rip_correction_milliseconds",  #
                108: "1/k0_pre_rip",  #
                109: "k0_pre_rip",  #
                110: "snr_pre_rip",  #
                111: "whm_pre_rip",  #
                112: "res_power_pre_rip",  #
                114: "signal_rip_voltage",  #
                115: "signal_pre_rip_voltage",  #
                116: "rip/prerip",  #
                119: "fims",  #
                }


    @staticmethod
    def extract_ims_header(filename):
        """
        Extract header from mcc/ims file
        For ims header specification see:
            DOI 10.1007/s12127-008-0010-9
            DOI 10.1351/pac200173111765
        :param filename:
        :return:
        """
        header_line_attribute_map = MccImsMeasurement.create_ims_header_line_attribute_map()
        header = OrderedDict()

        with open(filename, "rb") as fh:
            for i, line in enumerate(fh):
                line_attribute = ""
                try:
                    line_attribute = header_line_attribute_map[i]
                except KeyError:
                    if i == 131:
                        return header
                    else:
                        pass
                # ignoring the bits that are not unicode encoded, usually in the comment section (line 15)
                str_line = line.decode("utf-8", "ignore").strip()
                if i < 5:
                    str_line = str_line[:40]
                # check whether we have a comment line
                is_empty_line = len(str_line) < 3
                if not is_empty_line:
                    split_line = str_line.split(",", maxsplit=3)
                    try:
                        header[line_attribute] = split_line[2]
                    except IndexError as ie:
                        print(ie, "File []".format(filename), "line {}".format(i),
                              "len(split_line) = {}".format(len(split_line)))

        return header


    @staticmethod
    def extract_ims_header_from_string_lis(string_lis):
        """
        For ims header specification see:
            DOI 10.1007/s12127-008-0010-9
            DOI 10.1351/pac200173111765
        :type string_lis: list(str())
        :param string_lis: list of lines from header
        :return:
        """
        if len(string_lis) > 130:
            raise ValueError("List is too big to be an IMS-header. Expected 130 lines, got %s" % len(string_lis))
        header_line_attribute_map = MccImsMeasurement.create_ims_header_line_attribute_map()
        header = OrderedDict()

        for i, line in enumerate(string_lis):
            line_attribute = ""
            try:
                line_attribute = header_line_attribute_map[i]
            except KeyError:
                if i == 130:
                    return header
                else:
                    pass
            # already decoded binary string
            str_line = line.strip()
            if i < 5:
                str_line = str_line[:40]
            # check whether we have a comment line
            is_empty_line = len(str_line) < 3
            if not is_empty_line:
                split_line = str_line.split(",", maxsplit=3)
                try:
                    if split_line[2]:
                        header[line_attribute] = split_line[2]
                    else:
                        header[line_attribute] = ""
                except IndexError as ie:
                    raise IndexError(str(ie), "line {} len(split_line) = {}".format(line, len(split_line)))
        return header

        # ,data type,IMS raw data,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        # ,version,VOCan - v2.7 (2013-04-10),,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        # ,template version,0.3,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        # ,AD-board type,usbADC3,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        # ,ser.-no.,BD18,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        #
        # ,date,11/11/15
        # ,time,07:43:18
        # ,file,BD18_1511110743_ims.csv
        #
        #
        # ,SAMPLE INFORMATION,
        #
        # ,sample type,sample
        # ,sample ID,candy
        # ,comment,SpiroScout; Halls
        # ,location,
        # ,location name,
        # ,height ASL / m,0
        # ,total data acquisition time / s,149.074
        #
        #
        # ,IMS - INFORMATION,
        #
        # ,operator,
        # ,operator name,
        # ,IMS,BD18
        #
        # ,K0 RIP positive / cm^2/Vs,2.06
        # ,K0 RIP negative / cm^2/Vs,2.22
        # ,polarity,positive
        # ,grid opening time / us,300
        #
        # ,pause / s,0
        # ,tD interval (corr.) / ms from,-0.150
        # ,tD interval (corr.) / ms to,49.846
        # ,1/K0 interval / Vs/cm^2 from,-0.00437
        # ,1/K0 interval / Vs/cm^2 to,1.45258
        # ,no. of data points per spectra,2500
        # ,no. of spectra,300
        # ,no. averaged spectra,5
        # ,baseline / signal units,2011
        # ,baseline / V,4.9505816
        # ,V / signal unit,0.0024617512
        #
        # ,drift length / mm,120.0
        # ,HV / kV,4.38
        # ,amplification / V/nA,0.0
        #
        # ,drift gas,synth. air
        # ,drift gas flow / mL/min,100
        # ,sample gas,synth. air
        # ,sample gas flow / mL/min,99
        # ,carrier gas,synth. air
        # ,carrier gas flow / mL/min,151
        # ,pre-separation type,MCC OV5
        # ,pre-separation T / deg C,40.0; OK
        # ,sample loop T / deg C,40.0; OK
        # ,sample loop volume / mL,10.0
        #
        # ,ambient T source,manual
        # ,ambient T / deg C,-9999.9
        # ,ambient T x^2,0.0
        # ,ambient T x^1,1.0
        # ,ambient T x^0,0.0
        # ,ambient T x^-1,0.0
        # ,ambient T x^-2,0.0
        # ,ambient p source,manual
        # ,ambient p / hPa,-9999.9
        # ,ambient p x^2,0.0
        # ,ambient p x^1,1.0
        # ,ambient p x^0,0.0
        # ,ambient p x^-1,0.0
        # ,ambient p x^-2,0.0
        #
        # ,6-way valve,auto
        #
        #
        # ,EXTERNAL SAMPLING CONTROL,
        #
        # ,control status,SpiroScout||-50000||-999999999||-500000||10
        # ,control zero / signal units,-9999
        # ,control zero / V,-9999.0
        # ,control threshold / signal units,-9999.0
        # ,control threshold / V,-9999.0
        # ,control threshold2 / signal units,-9999.0
        # ,control threshold2 / V,-9999.0
        # ,control sampling time / s,-9999.0
        # ,control variable,Flow
        # ,control dimension,a.u.
        # ,control x^2,0.0
        # ,control x^1,1.0
        # ,control x^0,0.0
        # ,control x^-1,0.0
        # ,control x^-2,0.0
        #
        #
        # ,STATISTICS,
        #
        # ,RIP detection,enabled
        # ,tD  (RIP corr.) / ms,16.658
        # ,1/K0 (RIP) / Vs/cm^2,0.48543692
        # ,K0 (RIP) / cm^2/Vs,2.0599999017715502
        # ,SNR (RIP),158.25623
        # ,WHM (RIP) / Vs/cm^2,0.013072828
        # ,res. power (RIP),37.13447
        #
        # ,tD  (preRIP corr.) / ms,16.126
        # ,1/K0 (preRIP) / Vs/cm^2,0.46993366
        # ,K0 (preRIP) / cm^2/Vs,2.1279599390094988
        # ,SNR (preRIP),6.792517
        # ,WHM (preRIP) / Vs/cm^2,0.03814256
        # ,res. power (preRIP),21.43739
        #
        # ,signal RIP / V,-8.062418
        # ,signal preRIP / V,-0.13539632
        # ,RIP / preRIP,0.055680335
        #
        #
        # ,Fims / cm^2/kV,34.31548

    @staticmethod
    def parse_raw_ims_measurement(filename):
        """ Read in a csv file of an mcc ims measurement and return a dictionary holding the attributes
            :param filename : str(): path / BytesIO / StringIO to/of csv file
            :return: dict() rv
        """
        from ..tools.tools import deduplicate_retention_times
        # extract retention times from each measurement run, which is in line 131 which starts with "\   , tR, 0.0, 0.453, 0.983, ..."
        # real labels starts in line 132 with "1/K0, tDcorr.\SNr, 0, 1, 2, 3, 4, 5,..."
        if isinstance(filename, (BytesIO, StringIO)) or hasattr(filename, "readlines"):
            f = filename
        else:
            f = open(filename, 'rb')
        inverse_reduced_mobility, tDcorr, intensities, header_str = [], [], [], []
        for i, line in enumerate(f.readlines()):
            # str_line = line.decode()
            str_line = line.decode("utf-8", "ignore").strip()  # just ignore the bits that are not unicode encoded,
            # weird format in old Mcc/Ims files ~2006 - lines end with ,
            # usually in the comment section (line 15)
            # header case
            if i <= 131:  # line 1 to 130 are considered the header
                # if str.startswith(str_line, '#'):
                if i <= 129:
                    # header case
                    header_str.append(str_line)
                elif i == 130:
                    if str_line.endswith(","):
                        str_line = str_line[:-1]
                    # retention time case
                    if str.startswith(str_line, '\\'):
                        try:
                            # only works for new format > year 2006
                            retention_times = [float(ele) for ele in str_line[9:].strip('\n').split(', ')]

                        except ValueError as ve:
                            # only works with older MccIMS format ~2006, one of many issues
                            retention_times = [float(ele) for ele in str_line[3:].strip().split(', ')[1:]]
                            # if both don't work let ValueError propagate

                        # they might not outright raise a value error from 2006, but have bad indices with
                        #   duplicate retention times - which needs to be handled, we use it as index
                        if len(retention_times) == len(np.unique(retention_times)):
                            pass
                        else:
                            print("Duplicate values in retention_times index - will shift values by 1 second until there's a 1 second gap")
                            # upshifting is les problematic than downshifting - as later peaks are more spread out anyways
                            # need to do retention time shift from duplicate index onwards...
                            # find first duplicate index
                            # repeat as long as we have duplicates
                            while (len(retention_times) != len(np.unique(retention_times))):
                                retention_times = deduplicate_retention_times(retention_times)

                    else:
                        raise ValueError(
                            "IMS format doesn't match expected definition of DOI 10.1007/s12127-008-0010-9."
                            + " Expected \\ in line %d, got %s." % (i, str_line[0]))
                elif i == 131:
                    # column labels case
                    if str.startswith(str_line, '1/K0'):
                        labels = str_line.replace('\\', '').split(', ')
                    else:
                        raise ValueError(
                            "IMS format doesn't match expected definition of DOI 10.1007/s12127-008-0010-9."
                            + " Expected 1/K0 in line %d, got %s." % (i, str_line[:3]))

            # data case
            else:
                if str_line.endswith(","):
                    str_line = str_line[:-1]
                if str.startswith(str_line, '#'):
                    pass
                else:
                    lis = str_line.split(', ')
                    inverse_reduced_mobility.append(float(lis[0]))
                    tDcorr.append(lis[1])
                    intensities.append(np.array(lis[2:], dtype=float))


        header = MccImsMeasurement.extract_ims_header_from_string_lis(header_str)

        # Align measurement so that the RIP is exactly at 0.485
        # get rip positions from measurements and shift spectra if necessary
        # extract rip from header if possible
        rip_position = 0.0
        if header.get('rip_detection', 'disabled') == 'enabled': # disabled is default value
            rip_position = float(header['1/k0_rip'])
        if not "{0:.3f}".format(rip_position) == "0.485":
            #  need to extract rip position from spectra if not specified in header or for shifting
            # take 99 spectra with highest retention times, here we have the lowest probability of getting VOCs in
            # get maximum value and median position
            intensities_arr = np.vstack(intensities)
            # we are still dealing with negative values..... so we need to take the minimum
            # we might lose a lot of precision in this step
            rip_index = int(np.median(np.argmin(intensities_arr[:, len(retention_times)-100:], axis=0)))
            rip_position = inverse_reduced_mobility[rip_index]

            shift_needed = float(decimal.Decimal("{0:.3f}".format(rip_position)) - decimal.Decimal("0.485"))
            # shift all spectra to match rip position of 0.485
            if shift_needed != 0.0:
                print("Shifting RIP by {} from {} to 0.485".format(shift_needed, rip_position))
                inverse_reduced_mobility = list(np.array(inverse_reduced_mobility, dtype=float) + shift_needed)

            # save new rip position in the header
            header['rip_detection'] = 'enabled'
            header['1/k0_rip'] = '0.485'

        rv = {'inverse_reduced_mobility': inverse_reduced_mobility,
              "tDcorr": np.array(tDcorr, dtype=float),
              "intensities": np.abs(np.array(intensities)),
              'csv_name': filename,
              'header': header,
              'retention_times': retention_times
              }
        return rv


    def normalize_by_intensity(self):
        """
        Transform absolute intensity values to relative one. Using maximum intensity (RIP) as reference.
        Especially important if comparing measurements of different devices.
        """
        self.max_peak_intensity = np.max(self.df.values)
        # divide by peak intensity
        self.df /= np.max(self.max_peak_intensity)
        self.is_raw = False


    def baseline_correction_rip_compensation(self):
        """
        Remove rip tailing from spectra
        Get lower quantile for each 1/k0 and subtract intensities from spectra to remove RIP.
        Rip increases baseline of all spectra. This makes separation of peaks from noise more difficult.
        :return:
        """
        #  possibly extract noise and add mean of noise (=mean of values where 1/r0 < 0.4)
        # baseline correction only produces something meaningful when using positive values
        # Extract the quantiles of each 1/K0 value
        quantiles = self.df.quantile(0.25, axis=1)
        # Apparently the quantiles are subtracted columnwise --> transpose the DataFrame
        current_df = self.df.T
        # Subtract the quantiles
        current_df -= quantiles
        # Set negative values equals zero
        current_df[current_df < 0] = 0
        # transpose the DataFrame back into the correct shape
        self.df = current_df.T
        self.is_raw = False


    def denoise_gaussian_filter(self, sigma=1):
        """Apply gaussian filter for smoothing.
        sigma defines the std deviation of the gaussian kernel
        """
        self.df = pd.DataFrame(ndimage.gaussian_filter(self.df, sigma=sigma), index=self.inverse_reduced_mobility,
                               columns=self.retention_times)


    def denoise_savitzky_golay_filter(self, window_length=9, poly_order=2):
        """
        Apply Savitzky-Golay filter for denoising.
        """
        self.df = pd.DataFrame(signal.savgol_filter(self.df, window_length, poly_order, axis=1), index=self.inverse_reduced_mobility, columns=self.retention_times)


    def denoise_noise_subtraction(self, cutoff_1ko_axis=0.4):
        """
        Subtract noise defined as mean of values smaller than cutoff_1ko_axis.
        Modifies self.df
        """
        noise = self.df[self.df.index <= cutoff_1ko_axis].as_matrix().mean()
        self.df -= noise
        self.df[self.df < 0] = 0
        self.is_raw = False


    def denoise_median_filter(self, kernel_size=9):
        """
        Apply Median filter for denoising.
        kernel_size defines size of window used in the filter
        """
        self.df = pd.DataFrame(signal.medfilt2d(self.df, kernel_size), index=self.inverse_reduced_mobility,
                               columns=self.retention_times)

    def denoise_crop_inverse_reduced_mobility(self, cutoff_1ko_axis=0.4):
        """
        Crop intensity matrix by given 1/ko value.
        """
        self.df = self.df[self.df.index > cutoff_1ko_axis]
        # Update self.inverse_reduced_mobility
        self.inverse_reduced_mobility = np.array(self.df.index)


    def denoise_discrete_wavelet_transform(self, wavelet=pywt.Wavelet('db8'), level_inverse_reduced_mobility=4, level_retention_time=2, mode='smooth'):
        """
        Apply a wavelet transformation to denoise and compress the data.

        Standard parameters as recommended in DOI: 10.1021/ac503857y
        First in the inverse reduced mobility dimension and afterwards in the retention time dimension
        :param wavelet: Wavelet object to use in decomposition
        :type wavelet: pywt.Wavelet
        :param level_inverse_reduced_mobility: Decomposition level in Inverse Reduced Mobility dimension must be >=0
        :type level_retention_time: int
        :param level_retention_time: Decomposition level in Inverse Reduced Mobility dimension must be >=0
        :type level_retention_time: int
        :return:

        """
        # Decompositon rowwise in in 1/K0 dimension
        index = self.df.index
        columns = self.df.columns
        data = self.df.values
        limit_to_decomposition_level_ivr = min(level_inverse_reduced_mobility,
                                               pywt.dwt_max_level(data.shape[1],
                                               filter_len=wavelet.dec_len))
        limit_to_decomposition_level_rt = min(level_retention_time,
                                              pywt.dwt_max_level(data.shape[0],
                                              filter_len=wavelet.dec_len))

        # work flow from http://jseabold.net/blog/2012/02/23/wavelet-regression-in-python/
        for row in range(data.shape[0]):
            coeffs_ivr = pywt.wavedec(data[row], wavelet=wavelet, level=limit_to_decomposition_level_ivr, mode=mode, axis=0)
            threshold = np.std(coeffs_ivr[-1]) * np.sqrt(2 * np.log2(data.shape[1]))
            denoised_ivr = coeffs_ivr[:]
            denoised_ivr[1:] = (pywt._thresholding.hard(i, value=threshold, substitute=0) for i in denoised_ivr[1:])
            # Reconstruction of the row in 1/K0 dimension

            # handle dimension mismatch - if we gain a datapoint - remove it
            new_row = pywt.waverec(denoised_ivr, wavelet=wavelet, mode=mode, axis=0)
            correct_len = data[row].shape[0]
            data[row] = new_row[:correct_len]

        # decomposition in RT dimension
        coeffs_rt = pywt.wavedec(data, wavelet=wavelet, level=limit_to_decomposition_level_rt,mode=mode, axis=1)
        threshold = np.std(coeffs_rt[-1]) * np.sqrt(2 * np.log2(data.shape[0]))
        denoised_rt = coeffs_rt[:]
        denoised_rt[1:] = (pywt._thresholding.hard(i, value=threshold, substitute=0) for i in denoised_rt[1:])
        # reconstruction of the compressed data in RT dimension
        data = pywt.waverec(denoised_rt, wavelet=wavelet, mode=mode, axis=1)
        # data = pywt.waverec(coeffs_rt, wavelet=wavelet, mode=mode, axis=1)
        data = np.where(data < 0, 0, data)
        correct_len = len(columns)
        return pd.DataFrame(data=data[:, range(correct_len)], index=index, columns=columns)


    def export_raw(self, outpath=None, use_buffer=False):
        """
        Save an original representation of an MccImsMeasurement
        :param outpath:
        :param use_buffer:
        :return:
        """
        if outpath is None and not use_buffer:
            raise ExportError("Neither path nor buffer specified for export.")
        buffer = self._export_raw_helper()
        if outpath:
            with open(outpath, "w") as fh:
                fh.write(buffer)
        if use_buffer:
            return buffer


    def _export_raw_helper(self):
        """
        Handle work for export_raw
        :return:
        """
        # print header and then the intensity matrix
        # sorted_keys = sorted(self.header.keys())
        # rows look like
        # #,<field_name>,<value>
        header = self.header
        attribute_map = MccImsMeasurement.create_ims_header_line_attribute_map()

        # line 130 contains retention times
        line_130 = "\   , tR, " + ", ".join(map(str, np.round(self.retention_times, decimals=3))) + "\n"
        # line 131 is header - holds
        line_131 = "1/K0, tDcorr.\SNr, " + ", ".join(map(str,range(len(self.retention_times)))) + "\n"

        # in original format intensities are negative, and are ints
        if len(self.tDcorr):
             rows = [", ".join(map(str, (irm, self.tDcorr[i], *np.asarray(intensities.values, dtype=int)*-1)))+"\n" for i, (irm, intensities) in enumerate(self.df.iterrows())]
        else:
            rows = [", ".join(map(str, (irm, i*0.02, *np.asarray(intensities.values, dtype=int)*-1)))+"\n" for i, (irm, intensities) in enumerate(self.df.iterrows())]

        buffer = StringIO()
        for k in range(130):
            attribute = attribute_map.get(k, "")
            val = header.get(attribute, "")
            if attribute:
                row = "#,{},{}\n".format(attribute, val)
            else:
                row = "#\n"
            buffer.write(row)

        buffer.write(line_130)
        buffer.write(line_131)
        buffer.writelines(rows)
        rv = buffer.getvalue()
        buffer.close()
        return rv


    def export_to_csv(self, directory="", outname=None, use_buffer=False, file_suffix="_preprocessed"):
        """
        Save dataframe as csv. Use original directory if not further specified
        outname can be path or buffer
        :return:
        """
        if use_buffer:
            buffer = StringIO()
            self.df.to_csv(buffer, sep='\t', float_format='%.5f', index=True, header=True, index_label="index")
            buffer.seek(0)
            output = buffer.getvalue()
            buffer.close()
            return output

        if outname is None:
            if directory:
                outname = f"{directory}{self.raw_filename.rsplit('/', maxsplit=1)[1][:-4]}{file_suffix}.csv"
            else:
                outname = f"{self.raw_filename[:-4]}{file_suffix}.csv"

            # another old mcc-ims ~2006 problem - the index might not be unique - but a duplicate
            # in the last element occurs
            if any(self.df.columns.duplicated()):
                if self.df.columns.duplicated()[-1]:
                    print(f"{self.filename}: duplicate column - replacing last column name")
                    # replacement = {}
                    # replacement[self.df.columns[-1]] = self.df.columns[-1] +1
                    columns_copy = self.df.columns.copy().values
                    columns_copy[-1] +=1
                    # it's a Float64Index
                    replacement_columns = pd.Float64Index(columns_copy)
                    self.df.columns = replacement_columns

            self.df.to_csv(outname, sep='\t', float_format='%.5f', index=True, header=True, index_label="index")
        else:
            raise ValueError("Buffer or filename required for export. Got {}{} instead.".format(type(outname), outname))
        print("Saving preprocessed measurement to {}".format(outname))


    @staticmethod
    def import_from_csv(filename, class_label=""):
        """
        Import dataframe and reconstruct retention_times and inverse_reduced_mobility
        :param filename:
        :return: `MccImsMeasurement`
        """
        df = pd.read_csv(filename, sep="\t", comment="#", index_col="index", dtype=float, )
        # index is float64
        # set index values as float - otherwise it will be strings - make sure for both axes
        try:
            df.index = np.asarray(df.index, dtype=float)
            df.columns = pd.Float64Index(np.asarray(df.columns, dtype=float))
            # t_df = df.T
            # t_df.index = np.asarray(t_df.index, dtype=float)
            # # t_df.index = np.arange(len(t_df.index), dtype=float)  # index is from 1 to len - usually in half seconds instead?
            # df = t_df.T
            # replacement_columns = pd.Float64Index(columns_copy)
            # self.df.columns = replacement_columns
        except ValueError as ve:
            add_str = f" in file: {filename}"
            raise ValueError(str(ve), add_str)

        if class_label:
            # check whether column names are as expected
            measurement = MccImsMeasurement(filename, init_df=df, class_label=class_label)
        else:
            measurement = MccImsMeasurement(filename, init_df=df)
        return measurement


    def get_description(self):
        """
        Return dictionary holding measurement name, sample_id, sample_type, class (default None), comment
        :return:
        """
        return {'measurement_name': self.filename,
                'sample_id': self.header.get('sample_id', ''),
                'sample_type': self.header.get('sample_type', ''),
                'class_label': self.class_label,
                'comment': self.header.get('comment', '')}


    def get_retention_times(self):
        return self.retention_times


    def get_header(self):
        return self.header



class PeakAlignmentResult(object):
    """
    Result of a Peak alignment
    """
    MINIMUM_INVERSE_REDUCED_MOBILITY_RADIUS = 0.015
    MINIMUM_RETENTION_TIME_RADIUS = 3.0


    def __init__(self, number_of_clusters, peak_ids, peak_values_dict, measurement_names, peak_coordinates, **kwargs):
        """
        Create a Peak AlignmentResult, independent of type
        :param number_of_clusters:
        :param peak_ids:
        :param peak_values_dict:
        :param measurement_names:
        :param peak_coordinates:
        :param kwargs:
        """
        self.dict_of_df = dict()
        self.number_of_clusters = number_of_clusters
        self.peak_coordinates = peak_coordinates  # holds coordinates of the peaks with radius values
        # Dataframe holding measurement name, sample_id, sample_type, class_label and comment of measurement, if available
        # only available if we have original pipeline with raw / preprocessed measurements, not if starting from pdrs
        # save parameters used for clustering from kwargs
        params_used_for_clustering = dict()
        params_used_for_clustering.update(**kwargs)
        self.params_used_for_clustering = params_used_for_clustering
        if self.params_used_for_clustering['peak_alignment_step'].name == 'K_MEANS_CLUSTERING':
            print("Warning: K_MEANS_CLUSTERING is not deterministic!")
        # columns are peak_ids
        # rows are measurements
        # data is filled with peak_values
        for peak_detection in peak_values_dict.keys():
            data = None
            expected_shape = (len(measurement_names), len(peak_ids))
            if isinstance(peak_values_dict[peak_detection], np.ndarray):
                # check shape of peak_values to determine whether we need to transpose
                if peak_values_dict[peak_detection].shape == expected_shape:
                    # no reshape required
                    data = peak_values_dict[peak_detection]
                elif peak_values_dict[peak_detection].shape == expected_shape[::-1]:
                    # need to transpose
                    data = peak_values_dict[peak_detection].T
                else:
                    data = None

            elif isinstance(peak_values_dict[peak_detection], list):
                # check shape of peak_values to determine whether vstack or hstack is required
                if len(peak_values_dict[peak_detection]) == expected_shape[0]:
                    data = np.vstack(peak_values_dict[peak_detection])
                elif len(peak_values_dict[peak_detection]) == expected_shape[1]:
                    data = np.array(peak_values_dict[peak_detection]).T
                else:
                    data = None

            else:
                data = None

            if data is None:
                raise ValueError(f"Shape of peak_values does not match, require either " +
                                 f"{(len(measurement_names), len(peak_ids))} or " +
                                 f"{(len(peak_ids), len(measurement_names))}. Got: " +
                                 f"{(len(peak_values_dict[peak_detection]), len(peak_values_dict[peak_detection][0]))}")
            else:
                assert data.shape == expected_shape
            self.dict_of_df[peak_detection] = pd.DataFrame(data=data, index=measurement_names, columns=peak_ids)


class BitPeakAlignmentResult(PeakAlignmentResult):
    """
    Result of a Peak alignment. Uses True or False values to describe whether peak is present
    """
    def __init__(self, dict_of_clustering_result, dict_from_measurement_to_int, measurements, **kwargs):
        # add Peak_ and leading zeros to make size of peak_ids uniform

        if 'threshold_inverse_reduced_mobility' in kwargs:
            threshold_inverse_reduced_mobility = kwargs.pop('threshold_inverse_reduced_mobility')
        else:
            threshold_inverse_reduced_mobility = \
                MccImsAnalysis.DEFAULT_PREPROCESSING_PARAMETERS[PeakAlignmentMethod.PROBE_CLUSTERING][
                    'threshold_inverse_reduced_mobility']
        if 'threshold_scaling_retention_time' in kwargs:
            threshold_scaling_retention_time = kwargs.pop('threshold_scaling_retention_time')
        else:
            threshold_scaling_retention_time = \
                MccImsAnalysis.DEFAULT_PREPROCESSING_PARAMETERS[PeakAlignmentMethod.PROBE_CLUSTERING][
                    'threshold_scaling_retention_time']

        x_steps, y_steps = MccImsAnalysis.compute_window_size(
            threshold_inverse_reduced_mobility=threshold_inverse_reduced_mobility,
            threshold_scaling_retention_time=threshold_scaling_retention_time
        )

        peak_ids = ["Peak_{0:0>4}".format(c_id) for c_id in np.arange(start=0, stop=x_steps.shape[0] * y_steps.shape[0], step=1, dtype=int)]
        # shape = (rows, cols)
        number_of_clusters = dict()
        # number_of_clusters = np.unique(df_of_clustering_result['cluster'])
        peak_values = dict()
        peak_coordinates = dict()
        peak_coordinates_dict = dict()
        peak_descriptions = pd.DataFrame([m.get_description() for m in measurements])
        # quickfix works quite well
        # measurement_names = list(dict_from_measurement_to_int.keys())
        measurement_names = sorted(dict_from_measurement_to_int)


        for clustering_result in dict_of_clustering_result.keys():
            peak_values[clustering_result] = []
            peak_coordinates[clustering_result] = []
            number_of_clusters[clustering_result] = np.unique(dict_of_clustering_result[clustering_result]['cluster'])
            # measurement_names = list(dict_from_measurement_to_int.keys())
            feature_matrix = np.zeros((len(measurement_names), len(peak_ids)), dtype=bool)
            # iterate column wise
            for column_index in range(feature_matrix.shape[1]):
                # if peak is detected
                if column_index in np.unique(dict_of_clustering_result[clustering_result]['cluster']):
                    group = dict_of_clustering_result[clustering_result][
                        dict_of_clustering_result[clustering_result].cluster == column_index]
                    current_column = feature_matrix[:, column_index]
                    measurement_int_labels = np.unique(group['measurement_name'])
                    # set current column at measurement int labels = True, at other places leave 0
                    # use boolean logic, or
                    # make a mask to set values to 1
                    current_column[measurement_int_labels] = 1
                    peak_values[clustering_result].append(current_column)
                    # a Peak coordinate has:
                    # peak_id, (comment), inverse_reduced_mobility, retention_time
                    # a radius_inverse_reduced_mobility and a radius_retention_time
                    center_x = np.mean(group['inverse_reduced_mobility'])
                    # use population std, series.std() will calculate a sample std with 1 degree of freedom
                    deviation_x = np.maximum(np.std(group['inverse_reduced_mobility'], ddof=0),
                                         self.MINIMUM_INVERSE_REDUCED_MOBILITY_RADIUS)
                    center_y = np.mean(group['retention_time'])
                    # problematic. If peaks centers perfectly allign, we get a radius of 0. Instead we apply a fixed
                    #   minimum radius
                    # deviation_y = np.std(grqoup['retention_time'], ddof=0)

                    deviation_y = np.maximum(np.std(group['retention_time'], ddof=0),
                                             self.MINIMUM_RETENTION_TIME_RADIUS)

                    peak_id = peak_ids[column_index]
                    peak_coordinates[clustering_result].append({'peak_id': peak_id,
                                             'inverse_reduced_mobility': center_x,
                                             'retention_time': center_y,
                                             'radius_inverse_reduced_mobility': deviation_x,
                                             'radius_retention_time': deviation_y})
                else:
                    current_column = feature_matrix[:, column_index]
                    peak_values[clustering_result].append(current_column)


            #peak_descriptions = pd.DataFrame([m.get_description() for m in measurements])
            # Select all entries in df_of_clustering_result that have the same
            # pass measurements to constructor - case in which we import peak detection results
            peak_coordinates_df = pd.DataFrame(peak_coordinates[clustering_result])
            peak_coordinates_dict[clustering_result] = peak_coordinates_df
        super().__init__(number_of_clusters=number_of_clusters, peak_ids=peak_ids, peak_values_dict=peak_values,
                         measurement_names=measurement_names, #peak_descriptions=peak_descriptions,
                         peak_coordinates=peak_coordinates_dict, **kwargs)


    def __repr__(self):
        return f"BitPeakAlignmentResult: of {list(self.dict_of_df.keys())}"


class FloatPeakAlignmentResult(PeakAlignmentResult):
    """
    Result of a Peak alignment. Uses float values from peak intensity values to describe peaks.
    Can be converted to a BitPeakAlignment result by using a threshold.

    Handle the case of multiple peak detection steps applied in parallel. Create a dict of peak alignment results.
    Clustering results are stored in a dict
    """
    def __init__(self, dict_of_clustering_result, dict_from_measurement_to_int, measurements, **kwargs):
        # add Peak_ and leading zeros to make size of peak_ids uniform
        # fix missing parameters for alignment using different grid params

        if 'threshold_inverse_reduced_mobility' in kwargs:
            threshold_inverse_reduced_mobility = kwargs.pop('threshold_inverse_reduced_mobility')
        else:
            threshold_inverse_reduced_mobility = \
            MccImsAnalysis.DEFAULT_PREPROCESSING_PARAMETERS[PeakAlignmentMethod.PROBE_CLUSTERING][
                'threshold_inverse_reduced_mobility']
        if 'threshold_scaling_retention_time' in kwargs:
            threshold_scaling_retention_time = kwargs.pop('threshold_scaling_retention_time')
        else:
            threshold_scaling_retention_time = \
            MccImsAnalysis.DEFAULT_PREPROCESSING_PARAMETERS[PeakAlignmentMethod.PROBE_CLUSTERING][
                'threshold_scaling_retention_time']

        x_steps, y_steps = MccImsAnalysis.compute_window_size(
                threshold_inverse_reduced_mobility=threshold_inverse_reduced_mobility,
                threshold_scaling_retention_time=threshold_scaling_retention_time
        )
        peak_ids = ["Peak_{0:0>4}".format(c_id) for c_id in np.arange(start=0, stop=x_steps.shape[0] * y_steps.shape[0],
                                                                      step=1, dtype=int)]
        peak_coordinates_dict = dict()
        peak_values = dict()
        peak_coordinates = dict()
        number_of_clusters = dict()

        # relied on measurements being initialized - needs to work with peak detection results imported
        measurement_names = sorted(dict_from_measurement_to_int)

        for pdm_name in dict_of_clustering_result.keys():
            peak_values[pdm_name] = []
            peak_coordinates[pdm_name] = []
            number_of_clusters[pdm_name] = np.unique(dict_of_clustering_result[pdm_name]['cluster'])
            feature_matrix = np.zeros((len(measurement_names), len(peak_ids)), dtype=float)

            # dict_of_clustering_result['VISUALNOWLAYER'].loc[dict_of_clustering_result['VISUALNOWLAYER'].cluster == 444, ['retention_time', 'inverse_reduced_mobility']]
            # iterate column wise
            for column_index in range(feature_matrix.shape[1]):
                # if peak is detected
                if column_index in np.unique(dict_of_clustering_result[pdm_name]['cluster']):
                    group = dict_of_clustering_result[pdm_name][dict_of_clustering_result[pdm_name].cluster == column_index]
                    current_column = feature_matrix[:, column_index]
                    measurement_int_labels = np.unique(group['measurement_name'])

                    # if a cluster is present in a measurement take the maximum intensity of all peaks to represent the cluster
                    for measurement in measurement_int_labels:
                        current_column[measurement] = group.loc[group.measurement_name == measurement, 'intensity'].max()
                    peak_values[pdm_name].append(current_column)

                    # a Peak coordinate has:
                    # peak_id, (comment), inverse_reduced_mobility, retention_time
                    # a radius_inverse_reduced_mobility and a radius_retention_time
                    # check whether coordinates are already set - if not determine radii and assign standard parameters if too small
                    radius_columns = ['radius_inverse_reduced_mobility', 'radius_retention_time']

                    if all([col_name in group.columns for col_name in radius_columns]):
                        deviation_x = group['radius_inverse_reduced_mobility'].values[0]
                        deviation_y = group['radius_retention_time'].values[0]
                    else:
                        # use population std, series.std() will calculate a sample std with 1 degree of freedom
                        deviation_x = np.maximum(np.std(group['inverse_reduced_mobility'], ddof=0),
                                      self.MINIMUM_INVERSE_REDUCED_MOBILITY_RADIUS)
                        deviation_y = np.maximum(np.std(group['retention_time'], ddof=0),
                                             self.MINIMUM_RETENTION_TIME_RADIUS)

                    if deviation_y > 30:
                        # check if the grid window is just large - or if we have trouble clustering
                        n_peak_id = peak_ids[column_index]
                        peak_id_coord_df = pd.DataFrame(MccImsAnalysis._compute_peak_id_coord_map(
                            threshold_inverse_reduced_mobility=threshold_inverse_reduced_mobility,
                            threshold_scaling_retention_time=threshold_scaling_retention_time,
                        )).T
                        assigned_window = peak_id_coord_df[peak_id_coord_df.index == n_peak_id]

                        if deviation_y <= assigned_window['radius_retention_time'][0]:
                            pass
                        else :
                            print(f"Deviation_y is {deviation_y} for {pdm_name} column {peak_ids[column_index]}")
                            # nothing missing here - hasn't occured in practice

                    center_x = np.mean(group['inverse_reduced_mobility'])
                    center_y = np.mean(group['retention_time'])
                    peak_id = peak_ids[column_index]

                    peak_coordinates[pdm_name].append({'peak_id': peak_id,
                                             'inverse_reduced_mobility': center_x,
                                             'retention_time': center_y,
                                             'radius_inverse_reduced_mobility': deviation_x,
                                             'radius_retention_time': deviation_y})
                else:
                    current_column = feature_matrix[:, column_index]
                    peak_values[pdm_name].append(current_column)

            # Select all entries in df_of_clustering_result that have the same
            peak_coordinates_df = pd.DataFrame(peak_coordinates[pdm_name])
            peak_coordinates_dict[pdm_name] = peak_coordinates_df
        super().__init__(number_of_clusters=number_of_clusters, peak_ids=peak_ids, peak_values_dict=peak_values,
                         measurement_names=measurement_names, #peak_descriptions=peak_descriptions,
                         peak_coordinates=peak_coordinates_dict, **kwargs)


    def __repr__(self):
        return f"FloatPeakAlignmentResult: of {list(self.dict_of_df.keys())}"


class Analysis(object):
    """An analysis of a breath sample, composed of several Measurements
    """

    __metaclass__ = ABCMeta

    def __init__(self, measurements, preprocessing_steps, preprocessing_parameter_dict, performance_measure_parameter_dict, dataset_name, dir_level, class_label_file=""):
        """
        :type measurement: list(Measurement)
        :param measurement:
        :type preprocessing_steps: list(PreprocessingMethod)
        :param preprocessing_steps: list of PreprocessingMethod in order of execution:
        :type preprocessing_parameter_dict: dict(PreprocessingMethod : kwargs)
        :param preprocessing_parameter_dict: dict mapping PreprocessingMethod to parameters. Parameters passed will be overwriting standard parameters
        :type performance_measure_parameter_dict: dict(PerformanceMeasure : kwargs)
        :param performance_measure_parameter_dict: dict mapping PerformanceMeasure to parameters. Parameters passed will be overwriting standard parameters
        :type dataset_name: str
        :param dataset_name: String used as identifier and prefix for results
        :type dir_level: str
        :param dir_level: "" or "../" used to distinguish between different directory levels, depending on where the
                initial analysis script is called from.
        :param class_label_file: Name of class label file
        :type class_label_file: str
        """
        if measurements:
            if isinstance(measurements[0], Measurement):
                self.measurements = sorted(measurements, key=lambda m: m.filename if m.filename else m.raw_filename)
            else:
                self.measurements = sorted(measurements)
        else:
            self.measurements = measurements
        self.preprocessing_steps = preprocessing_steps
        self.dataset_name = dataset_name
        self.dir_level = dir_level  # is used for peax binary path and in saving overlap result
        self.preprocessing_parameter_dict = preprocessing_parameter_dict
        self.performance_measure_parameter_dict = performance_measure_parameter_dict
        self.class_label_file = class_label_file
        self.class_label_dict = {}
        if class_label_file:
            self.assign_class_labels()

    @abstractmethod
    def preprocess(self):
        """
        Apply preprocessing_steps
        """
        raise NotImplementedError()


    @staticmethod
    def execute_command(command_list, verbose=False):
        """
        Execute the command in separate subprocesses
        :param command_list: commandline arguments used to call peax with standard parameters
        :param verbose: flag to print the commandline arguments
        :return: triple of command used, exit code and error
        """
        try:
            if verbose:
                print("Starting command: ", command_list)
            return command_list, subprocess.check_call(command_list, stderr=subprocess.STDOUT), None
        except Exception as e:
            return command_list, None, e


    def assign_class_labels(self):
        """ 
        Read in label file set in `self.class_label_file` and assigns it to each measurement
        :return:
        """
        # check whether file exists, if not do not set class labels
        if Path(self.class_label_file).exists() or '.zip' in self.class_label_file:
            label_dict = Analysis.parse_class_labels(self.class_label_file)
        else:
            raise ValueError("Label File {} not found.".format(self.class_label_file))
        for m in self.measurements:
            m.set_class_label(label_dict.get(m.filename, None))
        self.class_labels = self.set_class_label_dict(label_dict)


    @staticmethod
    def check_for_peak_layer_file(archive_path):
        """
        Check archive path (zipfile) for *layer.csv
        :param archive_path:
        :return:
        """
        archive = zipfile.ZipFile(archive_path)
        potential_layers = [filename for filename in archive.namelist() if (str.endswith(filename, "layer.csv") or str.endswith(filename, "layer.xls"))]
        if potential_layers:
            return potential_layers[0]
        else:
            return ""


    @staticmethod
    def parse_peak_layer(visualnow_filename, memory_file=None):
        """
        Check if *layer.xls or *layer.csv is contained in zip archive and return pandas dataframe with expected columns
        :param archive: zip-archive
        :return:
        """
        if memory_file is not None:
            is_zip = False
            zip_split = visualnow_filename.split('.zip/')
            layer_name = zipfile.ZipFile(memory_file).open(zip_split[1],'r')
        else:
            is_zip = '.zip' in visualnow_filename
            if is_zip:
                zip_split = visualnow_filename.split('.zip/')
                if len(zip_split) == 2:
                    archive_name = zip_split[0] + '.zip'
                    layer_name = zipfile.ZipFile(archive_name).open(zip_split[1], 'r')
                else:
                    return None
            elif Path(visualnow_filename).exists() and (str.endswith(visualnow_filename, "layer.csv") or str.endswith(visualnow_filename, "layer.xls")):
                layer_name = visualnow_filename
            else:
                raise ValueError("Require absolute path to file names layer.csv or layer.xls or zip archive containing such file, not .".format(visualnow_filename))


        if visualnow_filename.endswith(".csv"):
            # read in the layer from a csv file
            peak_coord_df = pd.read_csv(layer_name, header=3)
        elif visualnow_filename.endswith(".xls"):
            # read in the layer from an excel file - need to buffer extra with new pandas version

            # extra check - in webplatform need extra buffer - in local version we let pandas do the reading
            if isinstance(layer_name, (str, Path)):
                peak_coord_df = pd.read_excel(io=layer_name, header=3)
            else:
                excel_buffer = BytesIO(layer_name.read())
                peak_coord_df = pd.read_excel(io=excel_buffer, header=3)
        else:
            raise ValueError("Layer file format not supported. Use csv or xls format.")
        if is_zip:
            layer_name.close()
        try:
            peak_coord_df.drop(['Comment'], axis=1, inplace=True)
            peak_coord_df.drop(['Color'], axis=1, inplace=True)
        except ValueError as ve:
            # if it doesnt have these columns, the better - we do a comparison in the next part
            pass
        except KeyError as ke:
            # if it doesnt have these columns, the better - we do a comparison in the next part
            pass

        visualnow_cols = ['1/K0', '1/K0 radius', 'RT', 'RT radius']
        using_vis_now_format = all(col_name in peak_coord_df.columns for col_name in visualnow_cols)

        own_cols = ['inverse_reduced_mobility', 'radius_inverse_reduced_mobility', 'retention_time', 'radius_retention_time']
        using_own_format = all(col_name in peak_coord_df.columns for col_name in own_cols)

        if not using_vis_now_format or using_own_format:
            raise InvalidLayerFileError("Not using expected columns. Either use {} or {}.".format(visualnow_cols, own_cols))

        # rename columns to match usual pattern or own pattern
        if using_vis_now_format:
            peak_coord_df.rename(columns={'1/K0': 'inverse_reduced_mobility',
                                      '1/K0 radius': 'radius_inverse_reduced_mobility',
                                      'RT': 'retention_time',
                                      'RT radius': 'radius_retention_time',
                                      'Name': 'peak_id'
                                      }, inplace=True)

        if ("peak_id" not in peak_coord_df.columns) or len(np.unique(peak_coord_df.peak_id)) != peak_coord_df.shape[0]:
            # assign peak_ids from 0 to coordinates.shape[0] -1
            peak_coord_df.peak_id = np.arange(0, peak_coord_df.shape[0], dtype=int)

        # only return columns we need
        return peak_coord_df[['peak_id', 'inverse_reduced_mobility', 'radius_inverse_reduced_mobility', 'retention_time', 'radius_retention_time']]


    @staticmethod
    def guess_class_label_extension(dir_to_search, file_list_alternative=[]):
        """
        Returns first instance of class_labels*[.csv, .tsv, .txt] found in dir_to_search
        :param dir_to_search:
        :return:
        """
        if ".zip" in str(dir_to_search) and not file_list_alternative:
            raise ValueError("Can't guess class_labels from zipfile")
        else:
            if file_list_alternative:
                file_list = file_list_alternative
                potential_class_label_file_names = [fn for fn in file_list if
                                                    Path(fn).stem.endswith("class_labels")]
            else:
                if not str(dir_to_search).endswith("/"):
                    dir_to_search = str(dir_to_search) + "/"
                file_list = Path(dir_to_search).glob("*")

                potential_class_label_file_names = [str(fn) for fn in file_list if fn.stem.endswith("class_labels")]
            supported_file_names = [fn for fn in potential_class_label_file_names if (fn.endswith(".csv") or fn.endswith(".tsv") or fn.endswith(".txt"))]

            # prioritize the "filtered_class_labels.csv" file
            priority_class_labels = [fn for fn in supported_file_names if fn.endswith("filtered_class_labels.csv")]
            if file_list_alternative:
                if priority_class_labels:
                    return priority_class_labels[0]
                else:
                    return supported_file_names[0]

            if priority_class_labels:
                try:
                    MccImsAnalysis.parse_class_labels(priority_class_labels[0])
                    return priority_class_labels[0]
                # keep going if we have multiple alternatives - dont just fail after find the _filtered csv if it's empty
                except (ValueError, IndexError):
                    pass

            for supported_file in supported_file_names:
                try:
                    MccImsAnalysis.parse_class_labels(supported_file)
                    return supported_file
                except (ValueError, IndexError):
                    pass
        raise ValueError(f"Couldn't find class label file in dir {dir_to_search}.")

            # label_dict = MccImsAnalysis.parse_class_labels("{}/{}".format(self.upload.path, class_label_file_name))
            # ims_filenames = [filename for filename in zip_content if str.endswith(filename, "_ims.csv")]
            # if '.zip/' in csv_filenames[0]:
            #     csv_filenames = [filename.split('.zip/')[1] for filename in csv_filenames]

    @staticmethod
    def parse_class_labels_from_ZipFile(archive, fn):
        """
        Create sorted `OrderedDict` from class label dict fn in archive
        :param archive:
        :param fn:
        :return:
        """
        if fn.endswith(".csv"):
            sep = ","
        elif fn.endswith(".tsv"):
            sep = "\t"
        elif fn.endswith(".txt"):
            sep = " "
        else:
            raise ValueError("Label file format not supported. Use csv or tsv format.")
        class_label_name = archive.open(fn)
        label_rows = pd.read_csv(class_label_name, delimiter=sep).values
        class_label_name.close()

        try:
            # its very likely this fails - so we need to make sure we only have 2 columns
            # if it has 3 columns we just assume someone also exported the index
            if len(label_rows[0]) == 2:
                dic = {fn: label for fn, label in label_rows}
            elif len(label_rows[0]) == 3:
                dic = {fn: label for _, fn, label in label_rows}
            else:
                raise ValueError(
                    "Problem parsing label file. Only .csv or .tsv supported. Please use only two columns in your file: name,class.")
            return OrderedDict(sorted(dic.items(), key= lambda t: t[0]))
        except ValueError as ve:
            raise ValueError("Problem parsing label file. Only .csv or .tsv supported. Please use only two columns in your file: name,class.")


    @staticmethod
    def parse_class_labels(filename):
        """
        Read in label file from filename and return dict from filename to class_label
        """
        # check whether file exists, if not do not set class labels
        is_zip = '.zip' in str(filename)
        if is_zip:
            archive = zipfile.ZipFile(filename.split('.zip')[0] + '.zip')
            class_label_name_archive = filename.rsplit('.zip/', maxsplit=1)[1]
            class_label_name = archive.open(class_label_name_archive, 'r')

        elif Path(filename).exists():
            class_label_name = filename
        else:
            raise ValueError("Label File {} not found.".format(filename))

        if filename.endswith(".csv"):
            sep = ","
        elif filename.endswith(".tsv"):
            sep = "\t"
        elif filename.endswith(".txt"):
            sep = " "
        else:
            raise ValueError("Label file format not supported. Use csv or tsv format.")

        label_rows = pd.read_csv(class_label_name, delimiter=sep).values

        if is_zip:
            class_label_name.close()
        try:
            # its very likely this fails - so we need to make sure we only have 2 columns
            # if it has 3 columns we just assume someone also exported the index
            if len(label_rows[0]) == 2:
                dic = {fn: label for fn, label in label_rows}
            elif len(label_rows[0]) == 3:
                dic = {fn: label for _, fn, label in label_rows}
            else:
                raise ValueError(
                    "Problem parsing label file. Only .csv or .tsv supported. Please use only two columns in your file: name,class.")
            # sort by key - so filename
            return OrderedDict(sorted(dic.items(), key=lambda t: t[0]))
        except ValueError as ve:
            raise ValueError("Problem parsing label file. Only .csv or .tsv supported. Please use only two columns in your file: name,class.")


    @abstractmethod
    def detect_peaks(self, peak_detection_method):
        """
        Use `peak_detection_method` to detect peaks
        :param peak_detection_method:
        :return:
        """
        raise NotImplementedError()


    @abstractmethod
    def align_peaks(self, peak_alignment_method):
        """
        Align or cluster peaks between measurements
        :type peak_alignment_method: PeakAlignmentMethod
        :return:
        """
        raise NotImplementedError()


    @abstractmethod
    def evaluate_performance(self, performance_measure):
        """
        Evaluate performance of all peak_detection methods according to metric
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def create_match_dicts(lis):
        """
        Match string representations with correct `PreprocessingMethod`
        :param lis:
        :return:
        """
        rv = {}
        for ele in lis:
            rv[ele] = ele
            rv[str(ele)] = ele
            rv[ele.name] = ele
        return rv


    @staticmethod
    def _prepare_parameter_kwargs_dict(dictionary, available_methods, default_params):
        """
        Fill dictionary with sensible default parameters for preprocessing steps
        Parameters in passed dictionary will be used to update all default values
        :param dictionary: dict containing parameter choices. Partial filled parameters are also possible
        :type return: dict()
        :return: dict()
        """
        # we need to map the string representation of the options to the actual method
        mapping_dict = Analysis.create_match_dicts(available_methods)

        # update default params for all options that have been chosen
        param_dict = {}
        for pre_k, v in dictionary.items():
            k = mapping_dict.get(pre_k, '')
            # check whether parameter is availble in default params
            if k and (k in default_params):
                # update / overwrite default with passed params
                param_copy = default_params[k].copy()
                param_copy.update(v)
                param_dict[k] = param_copy
            else:
                print("Removed {} from parameters, as no default available".format(k))
        return param_dict


    def is_able_to_cross_validate(self):
        """
        If more than 10 samples for minimum class - cross validation is possible
        :return:
        """
        if self.measurements:
            class_labels = [m.class_label for m in self.measurements]
        else:
            class_labels = self.class_label_dict.values()
        class_counts = Counter(class_labels)
        min_occurence = sorted(class_counts.values())[0]
        can_cross_validate = min_occurence >= 10
        return can_cross_validate

    def set_class_label_dict(self, class_label_dict):
        """
        Parse `class_label_dict` and set `self.class_label_dict`
        :param class_label_dict:
        :return:
        """
        # set class labels dict with Ordered Version
        self.class_label_dict = OrderedDict(sorted(class_label_dict.items(), key=lambda t: t[0]))

    def get_class_label_dict(self, class_label_fn=""):
        """
        Return `self.class_label_dict` - if not set try parsing it first
        :param class_label_fn:
        :return:
        """
        if self.class_label_dict:
            return self.class_label_dict
        else:
            if class_label_fn:
                dic = self.parse_class_labels(class_label_fn)
                self.set_class_label_dict(dic)
                return dic
            elif self.class_label_file:
                dic = self.parse_class_labels(self.class_label_file)
                self.set_class_label_dict(dic)
                return dic
            else:
                raise ValueError(f"Class label file not set - pass as argument or call parse_class_labels().")



class GCMSAnalysis(Analysis):
    """
    An GC-MS Analysis of a sample set - complementary to mcc-ims
    """

    def __init__(self, measurements, preprocessing_steps, dataset_name, dir_level,
                 preprocessing_parameters={}, performance_measure_parameters={}, class_label_file=""):
        """
        :param measurements: List of mzxml / mzml filenames
        :param preprocessing_steps: list of gcms peak detection methods
        :param preprocessing_parameter_dict:
        :param performance_measure_parameter_dict:
        :param dataset_name:
        :param dir_level:
        :param class_label_file:
        """
        preprocessing_dict = GCMSAnalysis.match_processing_options(preprocessing_steps)
        preprocessing_parameters = self.prepare_preprocessing_parameter_dict(preprocessing_parameters)


        super().__init__(measurements=measurements, preprocessing_steps=preprocessing_steps, dataset_name=dataset_name,
                         dir_level=dir_level, class_label_file=class_label_file,
                         preprocessing_parameter_dict=preprocessing_parameters,
                         performance_measure_parameter_dict=performance_measure_parameters,
                         )

        # self.normalization_steps = preprocessing_dict.get('normalization_steps', [])
        self.peak_detection_steps = preprocessing_dict.get('peak_detection_steps', [])
        # self.external_steps = preprocessing_dict.get('external_steps', [])
        self.peak_alignment_step = preprocessing_dict.get('peak_alignment_step', [])
        # self.denoising_steps = preprocessing_dict.get('denoising_steps', [])
        self.feature_xml_files = []
        self.feature_df = None

    @staticmethod
    def match_processing_options(options):
        """
        Match string instances to peak detection methods
        :return:
        """
        pd_map = Analysis.create_match_dicts(GCMSAnalysis.AVAILABLE_PEAK_DETECTION_METHODS)
        pa_map = Analysis.create_match_dicts(GCMSAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS)

        peak_detection_steps = [pd_map[step] for step in options if step in pd_map]
        peak_alignment_steps = [pa_map[step] for step in options if step in pa_map]

        if peak_alignment_steps:
            peak_alignment_step = peak_alignment_steps[0]
        else:
            peak_alignment_step = ""

        return {
            'peak_detection_steps': peak_detection_steps,
            'peak_alignment_step': peak_alignment_step,
        }

    @staticmethod
    def prepare_preprocessing_parameter_dict(dictionary):
        """
        Prepare prepocessing parameter dict with default parameters
        :param dictionary:
        :return:
        """
        return Analysis._prepare_parameter_kwargs_dict(dictionary,
                                                             available_methods=GCMSAnalysis.AVAILABLE_PREPROCESSING_METHODS,
                                                             default_params=GCMSAnalysis.DEFAULT_PREPROCESSING_PARAMETERS)

    @staticmethod
    def prepare_raw_filenames_class_labels(sample_dir, filelis=[]):
        """
        Prepare raw filenames (mzxml and mzml) and class label dict from sample_dir or zip.namelist()
        :param sample_dir:
        :param filelis:
        :return:
        """
        raw_files = filter_mzml_or_mzxml_filenames(sample_dir, filelis)
        # get class label dict from dir / zip - archive
        label_dict_path = GCMSAnalysis.guess_class_label_extension(sample_dir, filelis)
        label_dict = GCMSAnalysis.parse_class_labels(label_dict_path)
        return raw_files, label_dict

    def preprocess(self):
        """
        Preprocesing - not implemented - only applying wavelet transformation for raw MS1 spectra
        :return:
        """
        raise NotImplementedError("Not implemented, use `detect_peaks` instead")


    def detect_peaks(self):
        """
        Single core method for peak detection; Will apply raw/centroided peak detection depending on user selection
        :return:
        """
        feature_xmls = []
        # use parallel version if not debugging! - takes >=2 minutes per file
        for m_name in self.measurements:
            feature_xmls.append(GCMSAnalysis.detect_peaks_helper(
                    input_filename=m_name, peak_detection_method=self.peak_detection_steps[0],
                    preprocessing_parameter_dict=self.preprocessing_parameter_dict)['feature_storage'])

        self.feature_xml_files = sorted(feature_xmls)
        return feature_xmls


    def detect_peaks_parallel(self, num_cores=4):
        """
        Multi-core implementation of detect_peaks
        :return:
        """
        from multiprocessing import Pool

        # unpack params to make accessible for pool.map
        peak_detection_step = self.peak_detection_steps[0]
        preprocessing_parameter_dict = self.preprocessing_parameter_dict
        # tuples = [{
        #         "input_filename":m_name, "peak_detection_method":peak_detection_step,
        #         'preprocessing_parameter_dict': preprocessing_parameter_dict} for m_name in self.measurements]
        #
        tuples = [(m_name, peak_detection_step, preprocessing_parameter_dict) for m_name in self.measurements]

        with Pool(num_cores) as pool:
            result_params = pool.starmap(GCMSAnalysis.detect_peaks_helper, tuples)

        unsorted_feature_xml_fns = [result_param['feature_storage'] for result_param in result_params]
        self.feature_xml_files = sorted(unsorted_feature_xml_fns)
        return self.feature_xml_files


    @staticmethod
    def detect_peaks_helper(input_filename, peak_detection_method, preprocessing_parameter_dict={}):
        """
        Matches parameters for detect peaks. runs function for running preprocessing and peak detection.
        Distinguishing between raw and centroided
        :param input_filename:
        :param peak_detection_method:
        :param preprocessing_parameter_dict:
        :return:
        """
        helper_matcher = {
            GCMSPeakDetectionMethod.CENTROIDED: run_centroid_sample,
            GCMSPeakDetectionMethod.ISOTOPEWAVELET: run_raw_wavelet_sample,
        }
        parameters = preprocessing_parameter_dict.get(peak_detection_method, {})
        parameters['input_filename'] = input_filename
        return helper_matcher[peak_detection_method](parameters=parameters)

    @staticmethod
    def detect_peaks_helper_multicore(param_tup):
        """
        Matches parameters for detect peaks. runs function for running preprocessing and peak detection.
        Distinguishing between raw and centroided
        :return:
        """
        return GCMSAnalysis.detect_peaks_helper(*param_tup)


    def import_peak_detection_results(self, feature_xmls):
        """
        Import Feature_xmls for usage in `align_peaks` - will sort them alphabetically
        :param feature_xmls:
        :return:
        """
        print(f"Importing {len(feature_xmls)} Peak detection results")
        self.feature_xml_files = sorted(feature_xmls)


    def align_peaks(self, peak_alignment_method, save_to_disk=True):
        """
        Align peaks using methods from openms - resulting in FeatureMatrix ~ AnalysisResult / PeakAlignmentResult
        :param peak_alignment_method:
        :param consensus_outfile_name: where consensus file is saved to
        :type peak_alignment_method: GCMSAlignmentMethod
        :return:
        """
        print(f"Starting Alignment {peak_alignment_method.name}")
        peak_detection_method_name = self.peak_detection_steps[0].name

        result_data_dir = f"{self.dir_level}results/data/{self.dataset_name}/"
        # consensus export is now optional
        # consensus_outfile_name = f"{result_data_dir}{peak_alignment_method.name}.consensusXML"
        consensus_map, measurement_names = align_feature_xmls(
                feature_xml_lis=self.feature_xml_files, class_label_dict=self.get_class_label_dict())

        print(f"Converting Alignment to Feature Matrix")
        # print(self.performance_measure_parameter_dict)

        # TODO this should handle **kwargs instead - so matching by alignment_method and passing to make more extendable
        try:
            rounding_precision = self.performance_measure_parameter_dict[GCMSAlignmentMethod.POSE_FEATURE_GROUPER]['rounding_precision']
        except KeyError:
            rounding_precision = self.DEFAULT_PEAK_ALIGNMENT_PARAMETERS[GCMSAlignmentMethod.POSE_FEATURE_GROUPER]['rounding_precision']
        # use dict keys here aswell as when creating the consensus map - see align_feature_xmls
        feature_df = convert_consensus_map_to_feature_matrix(
            consensus_map, measurement_names=list(self.get_class_label_dict().keys()),
            rounding_precision=rounding_precision,
        )

        # if use_buffer:
        #     buffer = StringIO()
        #     self.feature_df = feature_df
        #     feature_df.to_csv(buffer, sep='\t', float_format='%.6f', index=True, header=True,
        #                           index_label="index")
        #     buffer.seek(0)
        #     output = buffer.getvalue()
        #     buffer.close()
        #     # out_list.append(output)

        if save_to_disk:
            self.feature_df = feature_df
            result_data_dir = f"{self.dir_level}results/data/{self.dataset_name}/"
            Path(result_data_dir).mkdir(parents=True, exist_ok=True)
            fn = "{}consensusXML_{}.csv".format(result_data_dir, peak_detection_method_name)
            print(f"Exporting Feature Matrix to disk {fn}")
            feature_df.to_csv(fn, sep='\t', float_format='%.6f', index=True, header=True,
                                  index_label="index")
            output = feature_df
        else:
            self.feature_df = feature_df
            output = feature_df

        return output

    @staticmethod
    def import_alignment_result(result_data_dir, peak_detection_method_name):
        """
        Import alignment result = FeatureMatrix with specific naming convention
        :param result_data_dir:
        :param peak_detection_method_name:
        :return:
        """
        fn = f"{result_data_dir}consensusXML_{peak_detection_method_name}.csv"

        fn_path = Path(fn)
        if not fn_path.exists():
            raise FileNotFoundError(fn_path)
        else:
            print(f"Importing Feature Matrix from disk {fn}")
            df = pd.read_csv(fn_path, sep='\t', index_col=0)  # using tsv even though csv name because reasons
            # use  sed -i 's/,/\t/g' consensusXML_ISOTOPEWAVELET.csv
            # to convert between , and \t
            return df.fillna(0.0)


    @staticmethod
    def split_matrix_train_test(full_matrix, train_class_label_dict, test_class_label_dict):
        """
        Split the `full_matrix` into train and test matrix based on the class_label dicts
        keys of class_label dicts must be used in index
        :param full_matrix:
        :param train_class_label_dict:
        :param test_class_label_dict:
        :return: train_matrix and test_matrix `list(pd.Dataframe)`
        """
        full_index = full_matrix.index
        train_mask = [fi in train_class_label_dict for fi in full_index]
        test_mask = [fi in test_class_label_dict for fi in full_index]

        train_matrix = full_matrix.loc[train_mask]
        test_matrix = full_matrix.loc[test_mask]
        return train_matrix, test_matrix


    def prepare_custom_fm_approach(self, feature_matrix):
        # feature_matrix = feature_matrix_model
        measurements = []
        outfile_names = []
        fileset_name = "custom_feature_matrix"

        preprocessing_steps = self.preprocessing_steps
        preprocessing_parameters = self.preprocessing_parameter_dict
        performance_measure_parameters = self.performance_measure_parameter_dict

        # plot_params, file_params = construct_default_parameters(fileset_name, folder_name="", make_plots=True)

        mcc_ims_analysis = MccImsAnalysis(measurements, preprocessing_steps=preprocessing_steps,
                                          outfile_names=outfile_names,
                                          preprocessing_parameters=preprocessing_parameters,
                                          dir_level=self.dir_level,
                                          dataset_name=self.dataset_name,
                                          class_label_file="",
                                          performance_measure_parameters=performance_measure_parameters
                                          )
        mcc_ims_analysis.set_class_label_dict(self.get_class_label_dict())

        #try to coerce PeakDetectionMethod - don't just default to CUSTOM
        coerced_pdm = self.peak_detection_steps[0]

        fm_dict = {coerced_pdm.name: feature_matrix}
        mcc_ims_analysis.analysis_result = AnalysisResult(peak_alignment_result=None, based_on_measurements=measurements,
                                                          peak_detection_steps=[coerced_pdm],
                                                          peak_alignment_step=self.peak_alignment_step, dataset_name=fileset_name,
                                                          class_label_dict=mcc_ims_analysis.get_class_label_dict(), feature_matrix=fm_dict)

        return mcc_ims_analysis

    AVAILABLE_PEAK_DETECTION_METHODS = (
        GCMSPeakDetectionMethod.CENTROIDED,
        GCMSPeakDetectionMethod.ISOTOPEWAVELET,
        GCMSPeakDetectionMethod.CUSTOM,
    )

    DEFAULT_PEAK_DETECTION_PARAMETERS = {
        GCMSPeakDetectionMethod.CENTROIDED: {},
        GCMSPeakDetectionMethod.ISOTOPEWAVELET: {"hr_data": False},
    }

    AVAILABLE_PEAK_ALIGNMENT_METHODS = (
        GCMSAlignmentMethod.POSE_FEATURE_GROUPER,
    )

    DEFAULT_PEAK_ALIGNMENT_PARAMETERS = {
        GCMSAlignmentMethod.POSE_FEATURE_GROUPER: {
            "rounding_precision": 0  # rounding precision for peak_id generation - number of decimals used
        }
    }

    AVAILABLE_PREPROCESSING_METHODS = tuple(chain(
        AVAILABLE_PEAK_DETECTION_METHODS,
        AVAILABLE_PEAK_ALIGNMENT_METHODS))

    DEFAULT_PREPROCESSING_PARAMETERS = {
        GCMSPeakDetectionMethod.ISOTOPEWAVELET: {  # no default for isotopewavelet - use default of openms
        },
        GCMSPeakDetectionMethod.CENTROIDED: {  # no default for centroided - use default of openms
        },
        GCMSPeakDetectionMethod.CUSTOM: {  # no default for CUSTOM
        },
        ### Alignment options also need default options
        GCMSAlignmentMethod.POSE_FEATURE_GROUPER: DEFAULT_PEAK_ALIGNMENT_PARAMETERS[GCMSAlignmentMethod.POSE_FEATURE_GROUPER],
    }


class MccImsAnalysis(Analysis):
    """
    An MCC-IMS Analysis of a breath sample set
    """


    def __init__(self, measurements, preprocessing_steps, outfile_names, dataset_name, dir_level, preprocessing_parameters={}, performance_measure_parameters={}, class_label_file="", visualnow_layer_file="", peax_binary_path=""):
        """
        :type measurements:  list(MccImsMeasurement)
        :param measurements: Measurements to be processed
        :type preprocessing_steps: list(MccImsPreprocessingMethod)
        :param preprocessing_steps: list of preprocessing steps applied to all measurements
        :type outfile_names: list(str)
        :param outfile_names: List of filenames used to save the peak_calling output to
        :type class_label_file: str
        :param class_label_file: Tab separated file holding the measurement names and associated class label
        :type visualnow_layer_file: str
        :param visualnow_layer_file: Excel table holding the peaks detected in VisualNow
        :type dataset_name: str
        :param dataset_name: String used as identifier and prefix for results
        :type dir_level: str
        :param dir_level: "" or "../" used to distinguish between different directory levels, depending on where the
                initial analysis script is called from.
        """
        # try casting them into preprocessing options
        preprocessing_dict = MccImsAnalysis.match_processing_options(preprocessing_steps)
        preprocessing_parameters = self.prepare_preprocessing_parameter_dict(preprocessing_parameters)
        performance_measure_parameters = self._prepare_performance_measure_parameter_dict(performance_measure_parameters)

        # first validate if preprocessing steps make sense and we have at least one peak detection method
        # and an alignment step
        if not (preprocessing_dict.get('external_steps', []) or
                    (preprocessing_dict.get('peak_detection_steps', []) and
                         preprocessing_dict.get('peak_alignment_step', []))):

            # only a problem when we have raw/processed measurements - not if reinitializing for feature matrix /
            #   custom peak_detection-results
            if measurements:
                raise ValueError("Need at least one external_peak_detection method or peak_detection method " +
                             "and one alignment method.")
        super().__init__(measurements=measurements, preprocessing_steps=preprocessing_steps, dataset_name=dataset_name,
                         dir_level=dir_level, class_label_file=class_label_file,
                         preprocessing_parameter_dict=preprocessing_parameters,
                         performance_measure_parameter_dict=performance_measure_parameters,
                         )
        self.normalization_steps = preprocessing_dict.get('normalization_steps', [])
        self.peak_detection_steps = preprocessing_dict.get('peak_detection_steps', [])
        self.external_steps = preprocessing_dict.get('external_steps', [])
        self.peak_alignment_step = preprocessing_dict.get('peak_alignment_step', [])
        self.denoising_steps = preprocessing_dict.get('denoising_steps', [])
        self.outfile_names = outfile_names
        self.peak_detection_results = dict()
        # self.peak_detection_results = []
        self.peak_alignment_result = None
        self.comparison_peak_detection_result = None
        self.analysis_result = None

        self.visualnow_layer_file = visualnow_layer_file

        self.peak_detection_combined = self.peak_detection_steps.copy()
        for external_step in self.external_steps:
            self.peak_detection_combined.append(external_step)

        self.peak_id_coord_map = None
        self.peax_binary_path = peax_binary_path
        # self._check_for_external_binaries_requirements()


    def __repr__(self):
        str_measurements = "\n\t"+"\n\t".join([str(m) for m in self.measurements])
        str_peak_detection_results = ""
        for peak_detection_method in self.peak_detection_results.keys():
            str_peak_detection_results += "\n{}:\n".format(peak_detection_method)
            str_peak_detection_results += ", ".join([str(pdr.measurement_name).split('_ims', maxsplit=1)[0]
                                                     for pdr in self.peak_detection_results[peak_detection_method]])
        str_peak_alignment_results = ""
        if self.peak_alignment_result:
            str_peak_alignment_results = "\n"
            for peak_detection_method in self.peak_alignment_result.dict_of_df.keys():
                str_peak_alignment_results += "For peak_detection_method {} we merged {} peaks in {} measurements with {}.\n".format(
                    str(peak_detection_method),
                    self.peak_alignment_result.dict_of_df[peak_detection_method].shape[1],
                    self.peak_alignment_result.dict_of_df[peak_detection_method].shape[0],
                    self.peak_alignment_step)
        return "Preprocessing Steps: {}\nMeasurements:{}\nPeak Detection Results:[{}]\n" \
               "Peak Alignment Results: {}\n".format(
            self.preprocessing_steps, str_measurements, str_peak_detection_results, str_peak_alignment_results)


    @staticmethod
    def match_processing_options(options):
        """
        Split options into preprocessing, denoising, peak-detection, external-peak-detection and peak-alignment options
        :param options:
        :return:
        """

        norm_map = Analysis.create_match_dicts(MccImsAnalysis.AVAILABLE_NORMALIZATION_METHODS)
        pd_map = Analysis.create_match_dicts(MccImsAnalysis.AVAILABLE_PEAK_DETECTION_METHODS)
        dn_map = Analysis.create_match_dicts(MccImsAnalysis.AVAILABLE_DENOISING_METHODS)
        ex_map = Analysis.create_match_dicts(MccImsAnalysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS)
        pa_map = Analysis.create_match_dicts(MccImsAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS)

        normalization_steps = [norm_map[step] for step in options if step in norm_map]
        peak_detection_steps = [pd_map[step] for step in options if step in pd_map]
        external_steps = [ex_map[step] for step in options if step in ex_map]
        denoising_steps = [dn_map[step] for step in options if step in dn_map]

        try:
            peak_alignment_step = [pa_map[step] for step in options if step in pa_map][0]
        except (KeyError, IndexError):
            peak_alignment_step = None

        return {'normalization_steps': normalization_steps,
                'peak_detection_steps': peak_detection_steps,
                'denoising_steps': denoising_steps,
                'external_steps': external_steps,
                'peak_alignment_step': peak_alignment_step,
        }


    @staticmethod
    def prepare_preprocessing_parameter_dict(dictionary):
        """
        Match preprocessing steps with default parameters and update `dictionary`
        :param dictionary:
        :return:
        """
        return MccImsAnalysis._prepare_parameter_kwargs_dict(dictionary,
                                            available_methods=MccImsAnalysis.AVAILABLE_PREPROCESSING_METHODS,
                                            default_params=MccImsAnalysis.DEFAULT_PREPROCESSING_PARAMETERS)


    def _prepare_performance_measure_parameter_dict(self, dictionary):
        """
        Match performance_measures with default parameters and update `dictionary`
        :param dictionary:
        :return:
        """
        return self._prepare_parameter_kwargs_dict(dictionary,
                                                   available_methods=MccImsAnalysis.AVAILABLE_EVALUATION_MEASURES,
                                                   default_params=MccImsAnalysis.DEFAULT_EVALUATION_MEASURE_PARAMETERS)


    def _check_for_external_binaries_requirements(self):
        """
        Check, whether all binaries for external steps are set if selected
        :return:
        """
        if self.external_steps:
            if ExternalPeakDetectionMethod.PEAX in self.external_steps:
                if not os.path.isfile(self.peax_binary_path):
                    raise PeakDetectionError("Binary for PEAX is not set up correctly.")
        if PeakDetectionMethod.VISUALNOWLAYER in self.peak_detection_steps:
            if not os.path.isfile(self.visualnow_layer_file):
                # maybe visaulnowfile is still in zip - disable default check ?
                raise PeakDetectionError("VisualnowLayer does not exist, please provide actual file.")


    def preprocess(self):
        """
        Apply preprocessing_steps to measurements
        First apply raw normalization steps, then PeakDetectionMethods and last PeakAlignmentMethods
        """
        # sanity check, whether binaries for external steps are set if selected
        # self._check_for_external_binaries_requirements()
        # for external steps apply external tool - external tools take raw measurements
        if self.external_steps:
            for external_step in self.external_steps:
                external_processing_map = {
                    ExternalPeakDetectionMethod.PEAX: self.external_peak_detection_peax,
                }
                # self.external_peak_detection_peax
                print(f"Applying External Preprocessing with {external_step}")
                external_processing_map[external_step]()
                # else:
                #     raise NotImplementedError("Choose PeaX or a VisualNow Layer as ExternalPeakDetectionMethod")
                # if we want to plot the intensity matrix we still need to normalize
        for measurement in self.measurements:
            measurement.normalize_measurement(self.normalization_steps)
            measurement.denoise_measurement(self.denoising_steps, param_map=self.preprocessing_parameter_dict)

        if isinstance(self.peak_detection_steps, list) and self.peak_detection_steps:
            for peak_detection_step in self.peak_detection_steps:
                self.peak_detection_results[peak_detection_step.name] = []
                self.detect_peaks(peak_detection_method=peak_detection_step)
        print("Finished preprocessing of measurements")

    def preprocess_multicore(self, num_cores=4):
        """
        Apply preprocessing steps to measurements using multiple cores
        First apply raw normalization steps, then PeakDetectionMethods and last PeakAlignmentMethods
        :param num_cores:
        :return:
        """
        # sanity check, whether binaries for external steps are set if selected
        # self._check_for_external_binaries_requirements()
        # for external steps apply external tool - external tools take raw measurements
        from multiprocessing import Pool

        with Pool(num_cores) as P:

            # address peax in multicore scenario
            if self.external_steps:
                for external_step in self.external_steps:
                    # external_processing_map = {
                    #     ExternalPeakDetectionMethod.PEAX: self.external_peak_detection_peax,
                    # }
                    # self.external_peak_detection_peax
                    print(f"Applying External Preprocessing with {external_step}")
                    ext_param_tuples = []

                    # create temporary directory for peax results
                    tmp_dir = os.path.join(tempfile.gettempdir(), '.breath/peax_raw/{}/'.format(hash(os.times())))
                    Path(tmp_dir).mkdir(parents=True)
                    for m in self.measurements:
                        ext_param_tuples.append((m, tmp_dir, self.peax_binary_path))

                    peax_pdr = P.map(MccImsAnalysis.external_peak_detection_peax_helper_multicore, ext_param_tuples)
                    # peax_pdr = [MccImsAnalysis.external_peak_detection_peax_helper_multicore(tup) for tup in ext_param_tuples]

                    # also add the PEAX pdrs
                    self.peak_detection_results[ExternalPeakDetectionMethod.PEAX.name] = peax_pdr

                    # cleanup - even if tempdir
                    rmtree(tmp_dir)

            # regular peak detection
            old_fns = np.array([m.filename for m in self.measurements])
            param_tuples = []
            for m in self.measurements:
                param_tuples.append((m, self.normalization_steps, self.denoising_steps, self.preprocessing_parameter_dict))
            new_measurements = P.map(MccImsAnalysis.preprocess_measurement, param_tuples)
            # new_measurements = [MccImsAnalysis.preprocess_measurement(param_tup) for param_tup in param_tuples]
            # need to check if order is still correct and need to resort
            new_fns = np.array([m.filename for m in new_measurements])
            print("Finished normalization and denoising")

            # Order is preserved
            # print(f"Order measurements is correct: {np.all(old_fns == new_fns)}")

            if isinstance(self.peak_detection_steps, list) and self.peak_detection_steps:
                # prepare tuple for mapping of params to cores
                pdm_param_tuples = []
                preprocessing_parameter_dict = MccImsAnalysis.prepare_preprocessing_parameter_dict(self.preprocessing_parameter_dict)

                # edge case - need to populate visualnow params if in peak_detection steps
                if PeakDetectionMethod.VISUALNOWLAYER in self.peak_detection_steps and not PeakDetectionMethod.VISUALNOWLAYER in preprocessing_parameter_dict:
                    preprocessing_parameter_dict[PeakDetectionMethod.VISUALNOWLAYER] = {"visualnow_filename": self.visualnow_layer_file}

                for new_m in new_measurements:
                    params=(new_m, self.peak_detection_steps, preprocessing_parameter_dict)
                    pdm_param_tuples.append(params)

                # create empty lists to hold peak detection results
                for peak_detection_step in self.peak_detection_steps:
                    self.peak_detection_results[peak_detection_step.name] = []

                print(f"Starting peak detection for {self.peak_detection_steps}")
                # use pool to do all internal peak detection in parallel
                # sorted by measurement - then in same order as self.peak_detection_steps
                pdrs = P.map(self.detect_peaks_helper_no_yield, pdm_param_tuples)
                # pdrs = [self.detect_peaks_helper_no_yield(tup) for tup in pdm_param_tuples]

                for pdrs_one_measurement in pdrs:
                    # has len(self.peak_detection_steps) many pdrs
                    # set self.peak_detection_results
                    for pdr in pdrs_one_measurement:
                        current_pdmn = pdr.peak_detection_step.name
                        self.peak_detection_results[current_pdmn].append(pdr)

                # print(pdrs)
                # print("#"*23)
                # print(self.peak_detection_results)
            print("Finished preprocessing of measurements")
        # Please make sure to save the preprocessed measurements in the analysis - otherwise we will work with raw ones later on
        self.measurements = new_measurements


    @staticmethod
    def preprocess_measurement(param_tuple):
        measurement, normalization_steps, denoising_steps, denoising_parameters = param_tuple
        measurement.normalize_measurement(normalization_steps)
        measurement.denoise_measurement(denoising_steps, param_map=denoising_parameters)
        return measurement



    def export_measurements(self, directory, buffer=None, file_suffix=""):
        """
        Export preprocessed files
        :return:
        """
        # no need for safety check that checks if it's actually still raw or not, will just overwrite
        rv = None
        if buffer is None:
            if file_suffix:
                for m in self.measurements: m.export_to_csv(directory=directory, file_suffix=file_suffix)
            else:
                for m in self.measurements: m.export_to_csv(directory=directory)
        else:
            rv = [m.export_to_csv(directory=directory, use_buffer=True, file_suffix=file_suffix) for m in self.measurements]
        return rv


    def export_preprocessed_measurements(self, directory, buffer=None):
        """
        Export all measurements -
        :param directory:
        :param buffer:
        :return:
        """
        return self.export_measurements(directory, buffer, file_suffix="_preprocessed")


    @staticmethod
    def read_preprocessed_measurements(measurement_paths):
        """
        Import preprocessed measurements
        :return:
        """
        measurements = [MccImsMeasurement.import_from_csv(ifn) for ifn in measurement_paths]
        return measurements


    @staticmethod
    def _extract_peak_intensities_for_coords_helper(measurement, coordinates):
        peak_intensities_per_measurement = []
        for _, row in coordinates.iterrows():
            # get all selection ranges from series i row
            center_irm = row['inverse_reduced_mobility']
            peak_id = row['peak_id']
            radius_irm = row['radius_inverse_reduced_mobility']
            radius_rt = row['radius_retention_time']
            center_rt = row['retention_time']

            irm_selection_window = center_irm - radius_irm, center_irm + radius_irm
            rt_selection_window = center_rt - radius_rt, center_rt + radius_rt

            # select by window
            window_by_irm = measurement.df.iloc[(measurement.df.index >= irm_selection_window[0]) & (measurement.df.index <= irm_selection_window[1])].T
            # further select from window matchig retention time - then get the max intensity
            mask_rt = (np.asarray(window_by_irm.index, dtype=float) >= rt_selection_window[0]) & (np.asarray(window_by_irm.index, dtype=float) <= rt_selection_window[1])
            max_intensity = np.max(window_by_irm.iloc[mask_rt].max())
            current_row = {'retention_time': center_rt, 'inverse_reduced_mobility': center_irm,
                           'intensity': max_intensity, 'measurement_name': measurement.filename,
                           'peak_id': peak_id
                           }
            peak_intensities_per_measurement.append(current_row)

        return pd.DataFrame.from_records(peak_intensities_per_measurement)


    @staticmethod
    def _extract_peak_intensities_for_coords(measurements, coordinates):
        assert isinstance(coordinates, pd.DataFrame)
        # check if peak_ids are unique in coordinates.peak_id column
        if ("peak_id" not in coordinates.columns) or len(np.unique(coordinates.peak_id)) != coordinates.shape[0]:
            # assign peak_ids from 0 to coordinates.shape[0] -1
            coordinates.peak_id = np.arange(0, coordinates.shape[0], dtype=int)

        peak_detection_dfs_of_measurements = []
        # iterate over all peak_coordinates
        for m in measurements:
            peak_detection_dfs_of_measurements.append(
                MccImsAnalysis._extract_peak_intensities_for_coords_helper(measurement=m, coordinates=coordinates))
        return peak_detection_dfs_of_measurements


    @staticmethod
    def detect_peaks_visualnow_helper(measurement, visualnow_df):
        return PeakDetectionResult(based_on_measurement=measurement, peak_df=MccImsAnalysis._extract_peak_intensities_for_coords_helper(measurement=measurement, coordinates=visualnow_df), peak_detection_method=PeakDetectionMethod.VISUALNOWLAYER)


    def read_in_from_visualnow_layer(self, **kwargs):
        """
        Reads in a Layer created by the software VisualNow. The Layer holds the central inverse reduced mobility and
        retention time coordinates as well as their radius.
        Based on the coordinates the intensities can be extracted from the measurement itself.
        :return: PeakDetectionResult
        """
        print('Loading Peak Layer from VisualNow')
        peak_detection_method = PeakDetectionMethod.VISUALNOWLAYER
        # read in the layer from an excel file
        # peak_coord_df = pd.read_excel(io=self.visualnow_layer_file, header=3)
        peak_coord_df = MccImsAnalysis.parse_peak_layer(self.visualnow_layer_file)

        peak_intensity_df_per_measurement = MccImsAnalysis._extract_peak_intensities_for_coords(self.measurements, peak_coord_df)

        results = [PeakDetectionResult(based_on_measurement=m,
                                       #outname=outname,
                                       peak_df=peak_df, peak_detection_method=peak_detection_method,
                                       )
                   for peak_df, m, outname in zip(
                peak_intensity_df_per_measurement, self.measurements, self.outfile_names)]
        self.peak_detection_results[PeakDetectionMethod.VISUALNOWLAYER.name] = results


    @staticmethod
    def read_in_custom_peak_detection(zip_path, result_suffix="_peak_detection_result.csv", zipfile_handle=None, pdm=ExternalPeakDetectionMethod.CUSTOM):
        """
        Should be in_memory file - read in all csv files and create Peak Detection results
        :param zip_path: Path to zip-archive
        :return: dict(), list(PeakDetectionResult)
        """

        # doesnt work - do set of equality checks instead
        # if not isinstance(pdm, (ExternalPeakDetectionMethod, PeakDetectionMethod)):
        # get all possible peak detection methods
        all_pdms = list(MccImsAnalysis.AVAILABLE_PEAK_DETECTION_METHODS)
        all_pdms.extend(list(MccImsAnalysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS))
        match_list = [pdm == a_pdm for a_pdm in all_pdms]
        if not any(match_list):
            raise ValueError(f"Need a PeakDetectionMethod - not {type(pdm)}")
        result_list = []
        if zipfile_handle is not None:
            candidate_lis = [fn for fn in zipfile_handle.namelist() if fn.endswith(result_suffix)]
            for fn in candidate_lis:
                result_df = PeakDetectionResult.import_from_csv(zipfile_handle.open(fn))
                raw_measurement_name = result_df.measurement_name[0]
                pdr = PeakDetectionResult(
                    based_on_measurement=raw_measurement_name,
                    peak_df=result_df, import_from_csv=True,
                    peak_detection_method=pdm)
                result_list.append(pdr)
                # archive.close(fn)
            class_label_file_name = MccImsAnalysis.guess_class_label_extension(dir_to_search=zip_path,
                                                                               file_list_alternative=zipfile_handle.namelist())
            label_dict = MccImsAnalysis.parse_class_labels_from_ZipFile(zipfile_handle, class_label_file_name)
        else:
            with zipfile.ZipFile(zip_path) as archive:
                candidate_lis = [fn for fn in archive.namelist() if fn.endswith(result_suffix)]
                for fn in candidate_lis:
                    result_df = PeakDetectionResult.import_from_csv(archive.open(fn))
                    raw_measurement_name = result_df.measurement_name[0]
                    pdr = PeakDetectionResult(
                        based_on_measurement=raw_measurement_name,
                        peak_df=result_df, import_from_csv=True,
                        peak_detection_method=pdm)
                    result_list.append(pdr)
                    # archive.close(fn)
                class_label_file_name = MccImsAnalysis.guess_class_label_extension(dir_to_search=zip_path, file_list_alternative=archive.namelist())
                label_dict = MccImsAnalysis.parse_class_labels_from_ZipFile(archive, class_label_file_name)
        result_list = sorted(result_list, key=lambda pdr_re: pdr_re.measurement_name)
        return label_dict, result_list


    @staticmethod
    def read_in_custom_feature_matrix(fn="", zip_path="", fm_suffix="_feature_matrix.csv", zipfile_handle=None):
        """
        Should be in_memory file - read in feature matrix csv file and class_label_dict
        :return: dict(), pd.DataFrame()
        """
        fm = None
        if not fn:
            if zipfile_handle is not None:
                fn = [filename for filename in zipfile_handle.namelist() if
                                        str.endswith(filename, fm_suffix)][0]

        if zipfile_handle is not None:
            with zipfile_handle.open(fn) as fm_fn:
                fm = pd.read_csv(fm_fn, index_col=0)
                class_label_file_name = MccImsAnalysis.guess_class_label_extension(dir_to_search=zip_path,
                                                                               file_list_alternative=zipfile_handle.namelist())
            label_dict = MccImsAnalysis.parse_class_labels_from_ZipFile(zipfile_handle, class_label_file_name)
        else:
            with zipfile.ZipFile(zip_path) as archive:
                if not fn:
                    fn = [filename for filename in archive.namelist() if
                          str.endswith(filename, fm_suffix)][0]

                with archive.open(fn) as fm_fn:
                    fm = pd.read_csv(fm_fn, index_col=0)
                # archive.close(fn)
                class_label_file_name = MccImsAnalysis.guess_class_label_extension(dir_to_search=zip_path,
                                                                                   file_list_alternative=archive.namelist())
                label_dict = MccImsAnalysis.parse_class_labels_from_ZipFile(archive, class_label_file_name)

        # sort feature matrix
        fm.sort_index(inplace=True)
        # result_list = sorted(result_list, key=lambda pdr: pdr.measurement_name)
        return label_dict, fm


    @staticmethod
    def external_peak_detection_peax_helper(raw_measurement, tempdir, peax_binary_path):
        # generate in and outname
        # if isinstance(raw_measurement, MccImsMeasurement):
        orig_name = raw_measurement.filename
        # in_name = orig_name.rsplit(".csv", maxsplit=1)[0] + "_in.csv"
        # had to remove _in suffix, as messing up class_label_dicts later on - was also encoded into df by peak
        in_name = Path(orig_name).stem + ".csv"

        out_name = orig_name.rsplit(".csv", maxsplit=1)[0] + "_out.csv"

        raw_export_path = tempdir + in_name
        outpath = tempdir + out_name
        # export raw file
        raw_measurement.export_raw(raw_export_path)

        command_list = [peax_binary_path, raw_export_path, outpath]
        # call peax
        # call_history = subprocess.check_call(command_list, stderr=subprocess.STDOUT), None
        # don't redirect stdout - celery will receive a sigpipe error as it also tries to write logs
        peax_dir = Path(peax_binary_path).parent
        # call peax
        # call_history = subprocess.check_call(command_list, stderr=subprocess.STDOUT), None
        # don't redirect stdout - celery will receive a sigpipe error as it also tries to write logs
        call_history = subprocess.check_call(command_list, cwd=peax_dir), None

        # call_history = MccImsAnalysis.peax_files(peax_binary_path, [raw_export_path], [outpath])
        if "Error" in call_history:
            raise PeakDetectionError(
                "An Error occured while using PEAX. Please make sure the binary is setup correctly")
        if -6 in call_history:
            raise PeakDetectionError(
                "An Error occured while using PEAX. PEAX con only process raw MCC-IMS files.")

        # reimport and return
        peak_df = PeakDetectionResult.import_from_csv(outpath)
        return PeakDetectionResult(
                based_on_measurement=raw_measurement,
                peak_df=peak_df, import_from_csv=True,
                peak_detection_method=ExternalPeakDetectionMethod.PEAX)

    @staticmethod
    def external_peak_detection_peax_helper_multicore(params):
        raw_measurement, tempdir, peax_binary_path = params
        return MccImsAnalysis.external_peak_detection_peax_helper(raw_measurement, tempdir, peax_binary_path)

    def external_peak_detection_peax(self):
        """
        Detect peaks with PEAX
        :return:
        """
        peak_detection_step = ExternalPeakDetectionMethod.PEAX
        # set this at startup of the project
        peax_binary_path = self.peax_binary_path
        infilenames = [m.raw_filename for m in self.measurements]

        tmp_dir = os.path.join(tempfile.gettempdir(), '.breath/peax_raw/{}'.format(hash(os.times())))
        have_created_temp_dir = any([".zip/" in fn for fn in infilenames])

        if have_created_temp_dir:
            infilenames = []
            os.makedirs(tmp_dir)
            # problem is, that we need actual raw measurements
            # - if we applied another peak detection method our measurements might have already been preprocessed
            # infile is a problem - peax cannot read from zip and they are not properly processed at prediction level
            # export in raw format to tmpdir
            for m in self.measurements:
                export_fn = m.raw_filename.rsplit(".zip/", maxsplit=1)[1]
                export_path = tmp_dir + export_fn
                m.export_raw(export_path)
                infilenames.append(export_path)

        outfilenames = self.outfile_names
        # check if files already exist - no need to do it twice
        are_all_files_already_peaxed = all([Path(on).exists() for on in outfilenames])
        if not are_all_files_already_peaxed:
            call_history = self.peax_files(peax_binary_path, infilenames, outfilenames)
            if "Error" in call_history:
                raise PeakDetectionError(
                    "An Error occured while using PEAX. Please make sure the binary is setup correctly")
            if -6 in call_history:
                raise PeakDetectionError(
                    "An Error occured while using PEAX. PEAX con only process raw MCC-IMS files.")
            PeakDetectionResult.peak_detection_result_suffix = "_ims_out"

        # need to read in csv files after peak detection!
        self.import_results_from_csv_list(outfilenames, peak_detection_step=peak_detection_step)

        if have_created_temp_dir:
            # clean up tempdir
            rmtree(tmp_dir, ignore_errors=True)

        PeakDetectionResult.peak_detection_result_suffix = PeakDetectionResult._peak_detection_original_result_suffix


    @staticmethod
    def peax_files(path_to_peax_binary, in_filenames, out_filenames, standard_parameters=True):
        """
        Uses Peax to identify peaks in the csv files and saves the predictions in the specified folder
        :param in_filenames: path of csv files to use as input for peax
        :param out_filenames: path of csv files to write to
        :param standard_parameters: use parameters given in parameters.cfg
        :return: a list of all output and error codes together with used parameters
        """
        command_list = []
        for in_file, out_file in zip(in_filenames, out_filenames):
            command_list.append([path_to_peax_binary, in_file, out_file])

        # need to copy parameters.cfg to current exectution dir
        base_config_path = Path(path_to_peax_binary).parent/"parameters.cfg"
        current_dir_path = Path(os.getcwd())/"parameters.cfg"
        copyfile(base_config_path, current_dir_path)

        p = multiprocessing.Pool(1)
        # save all output and error codes in a list
        # configuration file needs to be in the same directory as the execution level of the python script
        # - bad parsing from PEAX, we cannot pass arguments directly and it doesnt accept another path with -p as per help text
        # check current dir with os.getcwd() , and copy parameters.cfg file there
        call_history = [(used_command, output, error) for used_command, output, error in
                        p.map(Analysis.execute_command, command_list)]  # provide all commands to execute
        return call_history


    @staticmethod
    def _peax_write_config_file(t):
        tmp_conf = NamedTemporaryFile(delete=False)
        # write config file:
        conf_string = """
        # possible preprocessing modules: dn, s, bc
        preprocessing_first	bc
        preprocessing_second	dn
        preprocessing_third	s
        
        # possible candidate detection modules: lm, cf
        candidate_detection	cf
        
        # possible picking modules: ms, ce, emc
        # remark: using ce you have to download yoshiko 2.0 and copy binary in the same folder
        picking	emc
        
        # possible modeling modules: pme, e
        modeling	e
        
        tol_rt	3.0
        tol_rt_percent	0.1
        tol_rim	0.003
        intensity_threshold	10
        area_size	9
        fftcutoff	500
        expansion_size	10
        ce_weight_exponent	26
        """
        tmp_conf.write(conf_string)
        tmp_conf.close()
        return tmp_conf.name


    def detect_peaks(self, peak_detection_method):
        peak_detection_map = {
            PeakDetectionMethod.WATERSHED: self.detect_peaks_watershed,
            PeakDetectionMethod.TOPHAT: self.detect_peaks_tophat,
            PeakDetectionMethod.JIBB: self.detect_peaks_jibb,
            PeakDetectionMethod.VISUALNOWLAYER: self.read_in_from_visualnow_layer,
        }
        print("Applying Peak Detection {} with {}".format(peak_detection_method, self.preprocessing_parameter_dict.get(peak_detection_method, {})))
        peak_detection_map[peak_detection_method](**self.preprocessing_parameter_dict.get(peak_detection_method, {}))


    @staticmethod
    def detect_peaks_helper(measurement, peak_detection_steps, preprocessing_parameter_dict):
        helper_matcher = {
            PeakDetectionMethod.WATERSHED: MccImsAnalysis.detect_peaks_watershed_helper,
            PeakDetectionMethod.TOPHAT: MccImsAnalysis.detect_peaks_tophat_helper,
            PeakDetectionMethod.JIBB: MccImsAnalysis.detect_peaks_jibb_helper,
            PeakDetectionMethod.VISUALNOWLAYER: MccImsAnalysis.detect_peaks_visualnow_helper
        }
        for peak_detection_method in peak_detection_steps:
            if peak_detection_method == PeakDetectionMethod.VISUALNOWLAYER:
                yield helper_matcher[peak_detection_method](
                    measurement=measurement,
                    visualnow_df=MccImsAnalysis.parse_peak_layer(**preprocessing_parameter_dict.get(peak_detection_method, {})))
            else:
                yield helper_matcher[peak_detection_method](measurement=measurement, **preprocessing_parameter_dict.get(peak_detection_method, {}))

    @staticmethod
    def detect_peaks_helper_no_yield(param_tuple):
        measurement, peak_detection_steps, preprocessing_parameter_dict = param_tuple
        rv = []
        helper_matcher = {
            PeakDetectionMethod.WATERSHED: MccImsAnalysis.detect_peaks_watershed_helper,
            PeakDetectionMethod.TOPHAT: MccImsAnalysis.detect_peaks_tophat_helper,
            PeakDetectionMethod.JIBB: MccImsAnalysis.detect_peaks_jibb_helper,
            PeakDetectionMethod.VISUALNOWLAYER: MccImsAnalysis.detect_peaks_visualnow_helper
        }
        for peak_detection_method in peak_detection_steps:
            if peak_detection_method == PeakDetectionMethod.VISUALNOWLAYER:
                rv.append(helper_matcher[peak_detection_method](
                    measurement=measurement,
                    visualnow_df=MccImsAnalysis.parse_peak_layer(
                        **preprocessing_parameter_dict.get(peak_detection_method, {}))))
            else:
                rv.append(helper_matcher[peak_detection_method](
                    measurement=measurement,
                    **preprocessing_parameter_dict.get(peak_detection_method, {})))
        return rv



    @staticmethod
    def detect_peaks_jibb_helper(measurement, range_ivr=5, range_rt=7, noise_threshold=1.5):

        intensities = []

        intensity_threshold = 0.001
        # noise = measurement.df[measurement.df.index > 1.3].as_matrix().max()
        noise = intensity_threshold
        retention_time_lis = []
        inverse_reduced_mobility_lis = []
        # Intensity of a peak has to be greater/equal 3/2 of the noise
        threshold = noise_threshold * intensity_threshold
        # Create a intensity matrix which contains only intensities which are grater than the noise
        intensity_matrix = np.where(measurement.df > noise, measurement.df, 0)
        # Get the indices of the intensities which are greater than the noise
        inverse_reduced_mobility, retention_time = np.where(intensity_matrix > 0)
        # Put the indices into a dataframe. Column 0 --> inverse reduced mobility, column 1 --> retention time
        x = np.stack((inverse_reduced_mobility, retention_time), axis=1)
        x = pd.DataFrame(x)
        stepsize = 1
        # TODO: Get rid of the loops!
        # get the unique values of the inverse reduced mobility indices
        for i in np.unique(x[0]):
            # get a dataframe grouped by the inverse reduced mobility values
            groupedby_inverse_reduced_mobility = x[x[0] == i]
            # split the retention times in continuous groups
            consecutives_retention_time = np.split(groupedby_inverse_reduced_mobility[1].values,
                                                   np.where(np.diff(groupedby_inverse_reduced_mobility[1]) != stepsize)[
                                                       0] + 1)
            # Select only those splits which contains at least 15 signals
            consecutives_retention_time = [group for group in consecutives_retention_time if
                                           len(group) >= range_rt * 2 + 1]
            if len(consecutives_retention_time) == 0:
                pass
            else:
                for group_rt in consecutives_retention_time:
                    # Extract the index of the retention time with highest intensity
                    retention_time = group_rt[np.argmax(intensity_matrix[i, group_rt.min():group_rt.max()])]
                    """
                    Check, whether 7 signals before and after the signal with the max intensity increase or
                    decrease steadily, respectively, in retention time dimension
                    """
                    if np.all(np.ediff1d(
                            intensity_matrix[i, retention_time - range_rt: retention_time + 1]) > 0) and np.all(
                        np.ediff1d(intensity_matrix[i, retention_time: retention_time + range_rt + 1]) < 0):
                        # get a dataframe which is grouped by the extracted retention time
                        groupedby_retention_time = x[x[1] == retention_time]
                        # split the inverse reduced mobility into continuous groups
                        consecutives_inverse_reduced_mobility = np.split(groupedby_retention_time[0].values,
                                                                         np.where(np.diff(
                                                                             groupedby_retention_time[0]) != stepsize)[
                                                                             0] + 1)
                        # Select only those splits which contains at least 11 signals
                        consecutives_inverse_reduced_mobility = [group for group in
                                                                 consecutives_inverse_reduced_mobility if
                                                                 len(group) >= range_ivr * 2 + 1]
                        if len(consecutives_inverse_reduced_mobility) == 0:
                            break
                        else:
                            for group_irm in consecutives_inverse_reduced_mobility:
                                """
                                check if a group consists of at least 11 signals in inverse reduced mobility dimension
                                and check if the inverse reduced mobility value which you are checking for occurs in the group
                                """
                                if group_irm.max() < i:
                                    pass
                                elif i in group_irm:
                                    # Extract the index of the inverse reduced mobility with the highest intensity at the before extracted retention time
                                    inverse_reduced_mobility = group_irm[
                                        np.argmax(intensity_matrix[group_irm.min(): group_irm.max(), retention_time])]
                                    """
                                    Check, whether 5 signals before and after the signal with the maximal
                                    intensity increase or decrease steadily, respectively in inverse_reduced_mobility dimension
                                    """
                                    if np.all(np.ediff1d(intensity_matrix[
                                                         inverse_reduced_mobility - range_ivr: inverse_reduced_mobility + 1,
                                                         retention_time]) > 0) and np.all(np.ediff1d(
                                        intensity_matrix[
                                        inverse_reduced_mobility: inverse_reduced_mobility + range_ivr + 1,
                                        retention_time]) < 0):
                                        # Check if the intensity is above the threshold
                                        if intensity_matrix[inverse_reduced_mobility, retention_time] >= threshold:
                                            # safe the peak parameters
                                            inverse_reduced_mobility_lis.append(
                                                measurement.df.index[inverse_reduced_mobility])
                                            retention_time_lis.append(measurement.df.columns[retention_time])
                                            intensities.append(
                                                intensity_matrix[inverse_reduced_mobility, retention_time])
                                            break

                                        else:
                                            break
                                    else:
                                        break
                                elif group_irm.min() > i:
                                    break
                    else:
                        break
        # Create a dataframe which contains the peaks

        peak_df = MccImsAnalysis.form_peak_df(measurement.filename, retention_time_lis, inverse_reduced_mobility_lis, peak_names=np.arange(start=0, stop=len(inverse_reduced_mobility_lis), dtype=int), intensities=intensities)

        return PeakDetectionResult(based_on_measurement=measurement, peak_df=peak_df,  # out_name,
                                   peak_detection_method=PeakDetectionMethod.JIBB)


    def detect_peaks_jibb(self, range_ivr=5, range_rt=7, noise_threshold=1.5):
        """
        At least 5 intensity values above a certain threshold( 3/2 * noise) have to increase steadily in each dimension
        and after a  local maximum at least 5 intensity values above a certain threshold have to decrease steadily
        in each dimension to fulfill the requirements of being a peak.
        :return:
        """
        # TODO need to increase performance, right now by far slowest peak_detection with simplest filter
        for measurement in self.measurements:
            self.peak_detection_results[PeakDetectionMethod.JIBB.name].append(MccImsAnalysis.detect_peaks_jibb_helper(
                measurement=measurement, range_ivr=range_ivr, range_rt=range_rt, noise_threshold=noise_threshold))


    @staticmethod
    def detect_peaks_tophat_helper(measurement, noise_threshold=1.4):
        retention_times = []
        inverse_reduced_mobility_lis = []
        peak_names = []
        intensities = []

        intensity_threshold = 0.001
        # Calculate the threshold as the maximum value of the spectra with a inverse reduced mobility greater than 1.3

        # noise = noise_threshold * measurement.df[measurement.df.index > 1.3].as_matrix().mean()
        noise = noise_threshold * intensity_threshold

        # Apply TopHat
        # dataframe has no attribute dtype - probably want matrix instead - due to upgrade of scipy
        mask = white_tophat(measurement.df.values, footprint=np.ones((3, 3)))
        intensity_mask = np.where(mask > noise, mask, 0)
        # Extract the local maxima, based on the intensity values
        local_max = peak_local_max(intensity_mask, indices=True,
                                   footprint=np.ones((3, 3)))  # window to look for local maxima is a 3x3 matrix

        peak_id = 1
        # Get the peak coordinates and put them into a pd.DataFrame
        for coordinates in range(local_max.shape[0]):
            retention_time = measurement.df.columns[local_max[coordinates][1]]
            inverse_reduced_mobility = measurement.df.index[local_max[coordinates][0]]
            retention_times.append(retention_time)
            inverse_reduced_mobility_lis.append(inverse_reduced_mobility)
            peak_names.append(peak_id)
            intensities.append(measurement.df.loc[inverse_reduced_mobility][retention_time])
            peak_id += 1

        peak_df = MccImsAnalysis.form_peak_df(measurement.filename, retention_times, inverse_reduced_mobility_lis,
                                              peak_names, intensities)

        return PeakDetectionResult(based_on_measurement=measurement, peak_df=peak_df,  # out_name,
                                   peak_detection_method=PeakDetectionMethod.TOPHAT)


    def detect_peaks_tophat(self, noise_threshold=1.4):
        """
        Applies TopHat filtering to measurement.
        Sets all intensity values below a threshold to 0 and all intensity values above threshold to 1 for creating a
        mask.
        Further it creates a PeakDetectionResult for plotting the mask
        :return:
        """
        for measurement in self.measurements:
            self.peak_detection_results[PeakDetectionMethod.TOPHAT.name].append(
                MccImsAnalysis.detect_peaks_tophat_helper(measurement, noise_threshold))


    @staticmethod
    def detect_peaks_watershed_helper(measurement, noise_threshold=1.5):

        retention_times = []
        inverse_reduced_mobility_lis = []
        peak_names = []
        intensities = []

        """Background filtering"""
        # T, prep_meth = otsu_thresholding(intens_mat1)
        # T = np.mean(measurement.df)
        # T = 0
        """Calculate the euclidean distance for each point"""
        distance = ndimage.distance_transform_edt(
            measurement.df > np.mean(measurement.df.values) * noise_threshold)
        """Calculate the local maxima"""
        local_max = peak_local_max(distance, indices=False,
                                   footprint=np.ones((3, 3)))  # window to look for local maxima is a 3x3 matrix
        #                          threshold_abs = np.percentile(distance, 25)
        """Set the markers based on the local maxima"""
        markers, num_features = ndimage.label(local_max)
        """Calculate the peaks"""
        labels = watershed(distance, markers, mask=measurement.df.values)

        """Extract the coordinates of the peaks,
        the measurement_name and signal intensity and safe them as a csvfile"""
        for inverse_reduced_mobilities in range(labels.shape[0]):
            for retention_time in range(labels.shape[1]):
                if labels[inverse_reduced_mobilities][retention_time] == 0:
                    # non peaks don't get saved in the csv file
                    pass
                else:
                    retention_times.append(measurement.df.columns[retention_time])
                    inverse_reduced_mobility_lis.append(measurement.df.index[inverse_reduced_mobilities])
                    peak_names.append(labels[inverse_reduced_mobilities][retention_time])
                    intensities.append(measurement.df.values[inverse_reduced_mobilities][retention_time])

        peak_df = MccImsAnalysis.form_peak_df(measurement.filename, retention_times, inverse_reduced_mobility_lis, peak_names, intensities)

        return PeakDetectionResult(based_on_measurement=measurement, peak_df=peak_df,  # out_name,
                                             peak_detection_method=PeakDetectionMethod.WATERSHED)


    def detect_peaks_watershed(self, noise_threshold=1.5):
        """
        Detect the the peaks in a pandas DataFrame (intensity matrix) using watershed.
        Returns -- None - saves PeakDetectionResult in self.peak_detection_results[peak_detection_method_name]
        """
        # WATERSHED only runs on single core, which makes it a lot slower
        for measurement in self.measurements:
            self.peak_detection_results[PeakDetectionMethod.WATERSHED.name].append(
                MccImsAnalysis.detect_peaks_watershed_helper(measurement, noise_threshold=noise_threshold))


    @staticmethod
    def form_peak_df(measurement_name, retention_times, inverse_reduced_ion_mobility, peak_names, intensities):
        # is this a bottleneck? Could be written a lot more efficient
        # inverse_reduced_ion_mobility must be one dimensional - got weird list due to pandas index changes
        peak_df = pd.DataFrame({'retention_time': np.array(retention_times, dtype=float),
            'inverse_reduced_mobility': np.array(inverse_reduced_ion_mobility, dtype=float),
            'peak_id': peak_names,
            # same for intensity - got named tuples
            'intensity': np.array(intensities, dtype=float),
            'measurement_name': measurement_name})

        peak_df = peak_df.groupby(peak_df['peak_id'], as_index=False).mean()
        peak_df['retention_time'] = np.round(peak_df['retention_time'].values, decimals=3)
        # Convert the values into floats
        # peak_df['retention_time'] = peak_df['retention_time'].astype(np.float, copy=False)
        peak_df['inverse_reduced_mobility'] = peak_df['inverse_reduced_mobility'].astype(np.float, copy=False)
        return peak_df


    def align_peaks(self, file_prefix, alignment_result_type=FloatPeakAlignmentResult, **kwargs):
        """
        Align Peaks between measurements with clustering methods
        documentation for scikit-learn clustering methods: http://scikit-learn.org/stable/modules/clustering.html
        :param alignment_result_type: Defines the result type of the alignement
        :param kwargs: dictionary holding key value pairs for clustering algorithms
        :return:
        """
        peak_alignment_map = {
            PeakAlignmentMethod.K_MEANS_CLUSTERING: self._align_peaks_kmeans,
            PeakAlignmentMethod.DB_SCAN_CLUSTERING: self._align_peaks_dbscan,
            PeakAlignmentMethod.PROBE_CLUSTERING: self._align_peaks_probe_clustering,
            PeakAlignmentMethod.WARD_CLUSTERING: self._align_peaks_ward,
            PeakAlignmentMethod.WINDOW_FRAME: self._align_peaks_window_frame,
            PeakAlignmentMethod.AFFINITY_PROPAGATION_CLUSTERING: self._align_peaks_affinity_propagation,
            PeakAlignmentMethod.MEAN_SHIFT_CLUSTERING: self._align_peaks_mean_shift
        }
        print("Applying Peak Alignment {}".format(self.peak_alignment_step))

        self.peak_alignment_result = peak_alignment_map[self.peak_alignment_step](
            alignment_result_type=alignment_result_type,
            peak_alignment_step=self.peak_alignment_step,
            **self.preprocessing_parameter_dict.get(self.peak_alignment_step, {}))

        # is called after all peak detection and alignments are finished
        if len(self.peak_alignment_result.dict_of_df.keys()) >= 2 and len(self.peak_alignment_result.dict_of_df.keys()) < 5:
            self.overlap()
            # implement optimized overlap function, when necessary
            # self.new_overlap(self.peak_alignment_result.dict_of_df)


    def overlap(self):
        """
        Measure the overlap of different PeakDetectionMethods based on their PeakAlignmentResult.

        Assume the given points P(t,r) and Q(t,r) overlap, if |P(t) - Q(t)| < 0.015
        and |P(r) - Q(r)| < 3.000 + P(r) * 0.1.
        If a point in df1 overlaps at least with one point in df2 it is count as a
        overlap.
        """
        # why aren't we using sets of PeakIDs for comparison? shouldn't be too hard with intersection(set1, set2)
        print("Applying comparison of the applied PeakDetectionMethods")
        if len(self.peak_alignment_result.dict_of_df.keys()) > 4:
            raise NotImplementedError("VennDiagrams with more than 4 sets become too complex")

        # Merge all PeakALignmentResults to one big DataFrame
        df_of_all_keys = pd.concat([v for k,v in self.peak_alignment_result.dict_of_df.items()], keys=self.peak_alignment_result.dict_of_df.keys())
        df_of_all_keys = pd.DataFrame(df_of_all_keys.values, index=df_of_all_keys.index.droplevel(1))
        df_of_all_keys = df_of_all_keys.loc[:, (df_of_all_keys != 0).any(axis=0)]
        # Initialize an empty DataFrame, which will hold the overlaps
        overlap = pd.DataFrame(index=np.unique(df_of_all_keys.index.values),
                               columns=np.unique(df_of_all_keys.index.values))
        # Extract of amount of unique peaks which are detected by a certain PeakDetectionMethod
        for peak_detection in self.peak_alignment_result.dict_of_df.keys():
            overlap.loc[peak_detection, peak_detection] = np.unique(np.where(self.peak_alignment_result.dict_of_df[peak_detection])[1]).shape[0]

        # Check how many different PeakDetectionMethods have been applied
        if np.unique(df_of_all_keys.index.values).shape[0] == 4:
            overall_score = 0
            # Get the amount of unique peaks which all 4 PeakDetectionMethods have in common
            for peak in df_of_all_keys.columns.values:
                score = 0
                peak_of_interest = df_of_all_keys.loc[:, peak]
                for peak_detection in np.unique(df_of_all_keys.index.values):
                    if np.any(peak_of_interest.loc[peak_detection] != 0):
                        score += 1
                    else:
                        break
                if score == 4:
                    overall_score += 1
                    # Drop the peak to avoid multiple counting
                    df_of_all_keys.drop(peak, axis=1, inplace=True)
            # Assign the overlap to the DataFrame
            overlap.loc[:, 'all'] = int(overall_score)

            # Get the amount of unique peaks which 3 PeakDetectionMethods have in common
            peaks = []
            # all possible combination for 3 PeakDetectionMethods
            combinations = np.array([[0, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 0]], dtype=bool)
            # Set reference PeakDetectionMethod and the PeakDetectionMethods to compare with
            for peak_detection_reference in np.unique(df_of_all_keys.index.values):
                peak_detections_compare = [peak_detection for peak_detection in np.unique(df_of_all_keys.index.values)
                                           if peak_detection != peak_detection_reference]
                # Iterate over all possible combination
                for combination in combinations:
                    overall_score = 0
                    peak_detection_of_interest = sorted(list(compress(peak_detections_compare, combination)))
                    # Iterate over all potential peaks
                    for peak in df_of_all_keys.columns.values:
                        score = 0
                        peak_of_interest = df_of_all_keys.loc[:, peak]
                        if np.any(peak_of_interest.loc[peak_detection_reference] != 0):
                            for peak_detection in peak_detection_of_interest:
                                if np.any(peak_of_interest.loc[peak_detection] != 0):
                                    score += 1
                                else:
                                    break
                        if score == 2:
                            overall_score += 1
                            peaks.append(peak)
                    # Assign the overlap to the DataFrame
                    overlap.loc[peak_detection_reference, "{}_{}".format(peak_detection_of_interest[0],peak_detection_of_interest[1])] = overall_score
            # Drop the peaks to avoid multiple counting
            df_of_all_keys.drop(np.unique(peaks), axis=1, inplace=True)

            # Get the amount of unique peaks which 2 PeakDetectionMethods have in common
            peaks = []
            for peak_detection_reference in np.unique(df_of_all_keys.index.values):
                peak_detections_compare = [peak_detection for peak_detection in np.unique(df_of_all_keys.index.values)
                                           if peak_detection != peak_detection_reference]
                for peak_detection in peak_detections_compare:
                    score = 0
                    for peak in df_of_all_keys.columns.values:
                        peak_of_interest = df_of_all_keys.loc[:, peak]
                        if np.any(peak_of_interest.loc[peak_detection_reference] != 0) and np.any(
                                        peak_of_interest.loc[peak_detection] != 0):
                            score += 1
                            peaks.append(peak)
                    overlap.loc[peak_detection_reference, peak_detection] = score
            df_of_all_keys.drop(np.unique(peaks), axis=1, inplace=True)

        elif np.unique(df_of_all_keys.index.values).shape[0] == 3:
            overall_score = 0
            # Get the amount of unique peaks which all 3 PeakDetectionMethods have in common
            for peak in df_of_all_keys.columns.values:
                score = 0
                peak_of_interest = df_of_all_keys.loc[:, peak]
                for peak_detection in np.unique(df_of_all_keys.index.values):
                    if np.any(peak_of_interest.loc[peak_detection] != 0):
                        score += 1
                    else:
                        break
                if score == 3:
                    overall_score += 1
                    # Drop the peak to avoid multiple counting
                    df_of_all_keys.drop(peak, axis=1, inplace=True)
            # Assign the overlap to the DataFrame
            overlap.loc[:, 'all'] = int(overall_score)

            # Get the amount of unique peaks which 2 PeakDetectionMethods have in common
            peaks = []
            for peak_detection_reference in np.unique(df_of_all_keys.index.values):
                peak_detections_compare = [peak_detection for peak_detection in np.unique(df_of_all_keys.index.values)
                                           if peak_detection != peak_detection_reference]
                for peak_detection in peak_detections_compare:
                    score = 0
                    for peak in df_of_all_keys.columns.values:
                        peak_of_interest = df_of_all_keys.loc[:, peak]
                        if np.any(peak_of_interest.loc[peak_detection_reference] != 0) and np.any(
                                        peak_of_interest.loc[peak_detection] != 0):
                            score += 1
                            peaks.append(peak)
                    overlap.loc[peak_detection_reference, peak_detection] = score
            df_of_all_keys.drop(np.unique(peaks), axis=1, inplace=True)

        elif np.unique(df_of_all_keys.index.values).shape[0] == 2:
            # Get the amount of unique peaks which 2 PeakDetectionMethods have in common
            peaks = []
            for peak_detection_reference in np.unique(df_of_all_keys.index.values):
                peak_detections_compare = [peak_detection for peak_detection in np.unique(df_of_all_keys.index.values)
                                           if peak_detection != peak_detection_reference]
                for peak_detection in peak_detections_compare:
                    score = 0
                    for peak in df_of_all_keys.columns.values:
                        peak_of_interest = df_of_all_keys.loc[:, peak]
                        if np.any(peak_of_interest.loc[peak_detection_reference] != 0) and np.any(
                                        peak_of_interest.loc[peak_detection] != 0):
                            score += 1
                            peaks.append(peak)
                    overlap.loc[peak_detection_reference, peak_detection] = score
            df_of_all_keys.drop(np.unique(peaks), axis=1, inplace=True)

        # Get the amount of unique peaks which were only detected by on PeakDetectionMethod
        for peak_detection in np.unique(df_of_all_keys.index.values):
            peaks = []
            score = 0
            for peak in df_of_all_keys.columns.values:
                if np.any(df_of_all_keys.loc[peak_detection, peak] != 0):
                    score += 1
                    peaks.append(peak)
            overlap.loc[peak_detection, 'individual'] = score

            df_of_all_keys.drop(peaks, axis=1, inplace=True)
        assert df_of_all_keys.shape[1] == 0, "Not all peaks have been assigned to an intersection"

        self.comparison_peak_detection_result = overlap
        print("The overlaps of the different PeakDetectionMethods:\n{}".format(overlap))


    def _basic_clustering_method(self, func, **kwargs):
        """
        Apply clustering algorithm, follow the scipy structure. Save result to self.peak_alignment_result
        :param func: Clustering function producing a scipy.clustering like output. Allows generic functions to be passed
        :param func: clustering function producing a scipy.clustering like output
        :param kwargs: Key word arguments
        :return: None
        """
        # convert peak detection results to pd.DataFrame for convenience
        member_map = {}
        def assign_label(a_df, label):
            member_map[a_df['measurement_name'][0]] = label
            return a_df.assign(measurement_name=label)

        # get probe clustering grid params
        probe_alignment_params = self.preprocessing_parameter_dict.get(
                PeakAlignmentMethod.PROBE_CLUSTERING,
                MccImsAnalysis.DEFAULT_PREPROCESSING_PARAMETERS[PeakAlignmentMethod.PROBE_CLUSTERING])

        threshold_inverse_reduced_mobility = probe_alignment_params['threshold_inverse_reduced_mobility']
        threshold_scaling_retention_time = probe_alignment_params['threshold_scaling_retention_time']

        # Compute the winows for the standardized peak ids
        x_steps, y_steps = MccImsAnalysis.compute_window_size(
                threshold_inverse_reduced_mobility=threshold_inverse_reduced_mobility,
                threshold_scaling_retention_time=threshold_scaling_retention_time)

        def compute_peak_number(rt, irm):
            # get the index of the window on the inverse_reduced_mobility axis
            index_ivr = np.argmax(x_steps[x_steps <= irm]) + 1

            # get the index of the window on the retention time axis
            index_rt = np.argmax(y_steps[y_steps <= rt]) + 1

            # Compute the peak id
            peak_id = (index_rt - 1) * len(x_steps) + index_ivr
            return peak_id

        new_df_dict = dict()
        # map measurement_name from string to integer, otherwise fit() wont work
        for peak_detection in self.peak_detection_combined:
            # create dataframe from peak detection results
            df = pd.concat((assign_label(pdr.peak_df[['measurement_name', 'retention_time', 'inverse_reduced_mobility',
                                                      'intensity']], i) for i, pdr in enumerate(self.peak_detection_results[peak_detection.name])))
            df.reset_index(inplace=True)
            # Use Z-Normalization for the axes by dividing with standard deviation
            retention_std = df.retention_time.std()
            inverse_reduced_mobility_std = df.inverse_reduced_mobility.std()
            df.retention_time /= retention_std
            df.inverse_reduced_mobility /= inverse_reduced_mobility_std

            # fit using passed clustering function
            func.fit(df[['inverse_reduced_mobility', 'retention_time']])

            # assign cluster labels to df
            new_df = df.assign(cluster=func.labels_)
            new_df.retention_time *= retention_std
            new_df.inverse_reduced_mobility *= inverse_reduced_mobility_std


            # Substitute the assigned cluster numbers with standardized peak ids
            for column_index, group in new_df.groupby('cluster'):
                # handle the case if the cluster algorithm wasn't able to assing a peak to an cluster -> assign clusterid by grid but dont use mean of cluster

                if column_index == -1:
                    # eg in case of DBSCAN we need to assign labels for all outliers

                    for row_id, row in group.iterrows():
                        irm = row['inverse_reduced_mobility']
                        rt = row['retention_time']

                        # need to make work for application of DBSCAN - can be quite a lot of missed peaks -
                        # each of them should get their specific peak id as not part of any cluster
                        new_df.at[row_id, 'peak_id'] = compute_peak_number(rt, irm)

                else:
                    # a Peak coordinate has:
                    # peak_id, (comment), inverse_reduced_mobility, retention_time
                    # a radius_inverse_reduced_mobility and a radius_retention_time
                    center_x = np.mean(group['inverse_reduced_mobility'])
                    # use population std, series.std() will calculate a sample std with 1 degree of freedom
                    center_y = np.mean(group['retention_time'])

                    peak_id = compute_peak_number(center_y, center_x)

                    # print("cluster = {}, peak_id= {}".format(column_index, peak_id))
                    # overwrite the clusterid assigned by cluster alg with grid id
                    new_df.at[new_df.cluster == column_index, 'peak_id'] = peak_id

            # overwrite cluster columns with peak_id
            new_df_dict[peak_detection.name] = new_df.assign(cluster=np.array(new_df.peak_id.values, dtype=int))

        return new_df_dict, member_map, self.measurements


    @handle_peak_alignment_result_type
    def _align_peaks_dbscan(self, eps=0.025, min_samples=2):
        """
        Align peaks using the DBSCAN clustering method (see documentation http://scikit-learn.org/stable/modules/clustering.html#dbscan)
        :param eps:
        :param min_samples:
        :return:
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-2)
        return self._basic_clustering_method(dbscan, eps=eps, min_samples=min_samples)


    @handle_peak_alignment_result_type
    def _align_peaks_ward(self, **kwargs):
        raise NotImplementedError()


    @handle_peak_alignment_result_type
    def _align_peaks_spectral_clustering(self, affinity=0.3):
        raise NotImplementedError()


    @staticmethod
    def compute_window_size(threshold_inverse_reduced_mobility=0.015, threshold_scaling_retention_time=0.1, return_y_centers=False):
        """
        This function defines the edges of our clustering areas
        by defining the edges we can assign the cluster number faster
        With higher retention times, the peaks get blurry, therefore we linearly scale the space between clusters
        :param threshold_inverse_reduced_mobility:
        :param threshold_scaling_retention_time:
        :param return_y_centers:
        :return:
        """

        upper_y = 2000
        y_constant = 3.0
        # define the windows
        x_steps = np.linspace(0, 1.6, int(1.6 // (2 * threshold_inverse_reduced_mobility)))
        scaling_factor = 1 + threshold_scaling_retention_time

        def next_step(y):
            return scaling_factor*y + y_constant

        # retention time * threshold_scaling_retention_time + 3.0
        # retention time m
        y_steps = [0, 3.0]
        upper_limits = [0.0, 3.0]
        current_step = y_constant
        while current_step < upper_y:
            upper_limit = next_step(current_step)
            ns = next_step(upper_limit)
            upper_limits.append(upper_limit)
            y_steps.append(ns)
            current_step = ns

        if return_y_centers:
            return x_steps, np.asarray(upper_limits, dtype=float), y_steps
        else:
            return x_steps, np.asarray(upper_limits, dtype=float)


    @handle_peak_alignment_result_type
    def _align_peaks_probe_clustering(self, threshold_inverse_reduced_mobility=0.015, threshold_scaling_retention_time=0.1):
        """
        Align Peaks using the probe clustering algorithm (default)
        :param threshold_inverse_reduced_mobility:
        :param threshold_scaling_retention_time:
        :return:
        """
        print("Applying Probe clustering to {}".format(self.peak_detection_results.keys()))
        member_map = {}

        def assign_label(a_df, label):
            member_map[a_df['measurement_name'][0]] = label
            return a_df.assign(measurement_name=label)


        dict_of_df_peak_list = dict()

        x_steps, y_steps = MccImsAnalysis.compute_window_size(
            threshold_inverse_reduced_mobility=threshold_inverse_reduced_mobility,
            threshold_scaling_retention_time=threshold_scaling_retention_time,
        )

        def compute_peak_id(ivr, rt):
            # issue with negative ivr values - need to set index to zero if empty
            if not len(x_steps[x_steps <= ivr]):
                index_ivr = 0
            else:
                index_ivr = np.argmax(x_steps[x_steps <= ivr])
            # get the index of the window on the retention time axis
            index_rt = np.argmax(y_steps[y_steps <= rt])
            # Compute the peak id
            return index_rt * len(x_steps) + index_ivr

        # this relied on length of measurement list before - didnt work when reinitializing from peak detection results
        # num_of_measurements = len(self.measurements)
        one_pdmn = self.peak_detection_combined[0].name
        num_of_measurements = len(self.peak_detection_results[one_pdmn])

        # coord_map is computed if not already done previously
        peak_id_coord_df = pd.DataFrame(self.get_peak_id_coord_map(
            threshold_inverse_reduced_mobility=threshold_inverse_reduced_mobility,
            threshold_scaling_retention_time=threshold_scaling_retention_time,
        )).T
        peak_id_coord_df['cluster_no'] = peak_id_coord_df['cluster_no'].astype(int)

        peak_id_coord_df.reset_index(drop=True, inplace=True)

        # map measurement_name from string to integer, otherwise fit() wont work
        for peak_detection in self.peak_detection_combined:
            # concatenate all dataframes to a single one
            # for i, pdr in enumerate(self.peak_detection_results[peak_detection.name]):
            #     print("align_peaks_probe", i, peak_detection.name, pdr.shape)
            df_peak_list = pd.concat(
                (assign_label(
                    pdr.peak_df[['measurement_name', 'retention_time', 'inverse_reduced_mobility', 'intensity']], i)
                    for i, pdr in enumerate(self.peak_detection_results[peak_detection.name])))

            df_peak_list.reset_index(drop=True, inplace=True)

            # first we need to merge clusters that are overlapping according to peak_id computation
            # vectorized assignment of cluster_id
            df_peak_list['cluster'] = np.vectorize(compute_peak_id)(df_peak_list['inverse_reduced_mobility'].values, df_peak_list['retention_time'].values)

            # select grid coordinates from coord map - but order is not by cluster_no
            # select peak_ids that are used in current peak_detection
            wanted_cluster_nos = set(np.unique(df_peak_list['cluster']))
            coord_mask = [no in wanted_cluster_nos for no in peak_id_coord_df['cluster_no']]

            radii = peak_id_coord_df[coord_mask][['cluster_no', 'radius_retention_time', 'radius_inverse_reduced_mobility', 'retention_time', 'inverse_reduced_mobility']]

            result_list = []
            # merge all peaks within same cluster of same measurement
            for m_number in range(num_of_measurements):
                peaks_in_one_measurement = df_peak_list[df_peak_list['measurement_name'] == m_number]
                for cluster_id in np.unique(peaks_in_one_measurement['cluster']):
                    rows_matching_cluster = peaks_in_one_measurement[peaks_in_one_measurement['cluster'] == cluster_id]
                    row = np.mean(rows_matching_cluster).to_dict()

                    # we need to update radii
                    current_cluster_mask = radii['cluster_no'] == cluster_id

                    # fixed non existent values when clustering with non-default probe clustering params - coords were comkputed with defautls
                    r_rt, r_irm, rt, irm = radii[current_cluster_mask][[
                            'radius_retention_time', 'radius_inverse_reduced_mobility', 'retention_time',
                            'inverse_reduced_mobility']].values[0]

                    # overwrite retention time and irm with that of window
                    row.update({'radius_retention_time' : r_rt, 'radius_inverse_reduced_mobility': r_irm,
                                'retention_time': rt, 'inverse_reduced_mobility': irm})
                    result_list.append(row)

            df_peak_list = pd.DataFrame(result_list)
            # cast cluster column and measurement_name to int
            df_peak_list[['cluster', 'measurement_name']] = df_peak_list[['cluster', 'measurement_name']].astype(int)

            dict_of_df_peak_list[peak_detection.name] = df_peak_list
        return dict_of_df_peak_list, member_map, self.measurements


    @handle_peak_alignment_result_type
    def _align_peaks_window_frame(self, distance_threshold=(3, 0.02)):
        """
        Assign clusters based on fixed distance threshold on x and y axis
        :param distance_threshold: Threshold on (retention_time, inverse_reduced_mobility)
        :return:
        """
        # when adding new point, only add if threshold to first point is not breached
        delta_r, delta_t = distance_threshold

        # convert peak detection results to pd.DataFrame for convenience
        member_map = {}

        def assign_label(a_df, label):
            # we have problems when importing from peak_detection_results and external at the same time
            # here filenames will not match up --> problem when creating trainings matrix
            member_map[a_df['measurement_name'][0]] = label
            return a_df.assign(measurement_name=label)

        # map measurement_name from string to integer, otherwise fit() wont work
        df = pd.concat(
            (assign_label(pdr.peak_df[['measurement_name', 'retention_time', 'inverse_reduced_mobility']], i) for i, pdr
             in enumerate(self.peak_detection_results)))
        # df = pd.DataFrame(df[['retention_time', 'inverse_reduced_mobility']])
        # Normalize the scales of the axes by dividing with standard deviation
        df.retention_time /= df.retention_time.std()
        df.inverse_reduced_mobility /= df.inverse_reduced_mobility.std()

        df = df.assign(row_id=np.arange(0, df.shape[0], dtype=int))
        df.index = np.arange(0, df.shape[0], dtype=int)

        # distance to origin
        def compute_distance(a, b):
            return np.sqrt(np.square(a) + np.square(b))

        df['distance'] = np.vectorize(compute_distance)(df['retention_time'], df['inverse_reduced_mobility'])
        sorted_distance_index = np.argsort(df['distance'].copy().values)
        # manage filter list, which will be used to filter elements already included in solution from sorted distance index
        distance_filter = np.ones_like(sorted_distance_index, dtype=bool)

        # map new_df index to index in sorted_distance_index
        distance_index_remap = np.argsort(sorted_distance_index)

        # add cluster column to new_df
        df = df.assign(cluster=np.zeros(len(df), dtype=int))

        # get up to no_measurements many points in one cluster
        def check_rt_threshold_and_cluster(cr, ct, r, t, cluster):
            # check if peak is contained in window
            r_satisfied = (r <= cr + delta_r + 0.1 * cr) and (r >= cr - delta_r - 0.1 * cr)
            t_satisfied = (t <= ct + delta_t) and (t >= ct - delta_t)
            is_in_window = (r_satisfied and t_satisfied)

            # check if cluster has already been set
            has_been_added = cluster
            return is_in_window and not has_been_added

        cluster_number = 1
        # as long as there are non labelled peaks continue
        while np.any(distance_filter):
            # select all rows where cluster hasn't been assigned yet
            # select peak with lowest distance to 0 as new starting point
            current_peak_index = sorted_distance_index[distance_filter][0]
            current_peak = df.iloc[current_peak_index]
            # remove peak_id from possible neighbors
            distance_filter[current_peak_index] = False

            # find new cluster by adding neighbors in distance
            current_r = current_peak.retention_time
            current_t = current_peak.inverse_reduced_mobility

            neighbor_mask = np.vectorize(check_rt_threshold_and_cluster)(current_r, current_t, df['retention_time'], df['inverse_reduced_mobility'],
                                                                         df['cluster'])
            new_cluster_indices_list = list(df[neighbor_mask].index.values)
            new_cluster_indices_list.append(current_peak_index)

            df.loc[neighbor_mask, 'cluster'] = cluster_number

            # update distance_filter - aka remove labeled peaks from sorted list
            distance_filter[distance_index_remap[new_cluster_indices_list]] = False

            cluster_number += 1
        return df, member_map, self.measurements


    @handle_peak_alignment_result_type
    def _align_peaks_affinity_propagation(self, preference=-0.1):
        """
        Align Peaks using the affinity propagation algorithm
        :param preference:
        :return:
        """
        af = AffinityPropagation(preference=preference)
        return self._basic_clustering_method(af)


    @handle_peak_alignment_result_type
    def _align_peaks_mean_shift(self, bandwidth=0.1):
        """
        Align Peaks using the mean shift algorithm
        :param bandwidth:
        :return:
        """
        # non flat geometry estimation of bandwidth
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True, n_jobs=-2)
        return self._basic_clustering_method(ms)


    @handle_peak_alignment_result_type
    def _align_peaks_kmeans(self, n_clusters=80):
        """
        Align Peaks using the kmeans algorithm
        :param alignment_result_type:
        :param n_clusters:
        :return:
        """
        #   kmeans, see documentation http://scikit-learn.org/stable/modules/clustering.html#kmeans
        # create new Kmean clustering instance
        km = KMeans(n_clusters=n_clusters, n_jobs=-2)
        return self._basic_clustering_method(km)


    @staticmethod
    def remove_redundant_features_helper(bit_trainings_matrix, class_labels, percentage_threshold):
        # ! no longer or-ing with the inverse bin_trainings_matrix
        # we want to keep positive signals, not remove them!
        # only check low threshold
        return MccImsAnalysis._remove_redundant_features_helper(bit_trainings_matrix, class_labels, percentage_threshold)
            # MccImsAnalysis._remove_redundant_features_helper(np.logical_not(bit_trainings_matrix), class_labels, percentage_threshold)
            # )


    @staticmethod
    def _remove_redundant_features_helper(bit_trainings_matrix, class_labels, percentage_threshold):
        # creates a mask for column selection
        # check for presence of 1s in percentage of columns, columns which do not qualify will be False
        filter_mask_by_class = dict()
        # need to look at measurements classwise and apply threshold classwise
        # None type is a problem here - if measurements don't have an assigned type - make sure users get info, that we removed files from set due to not having a label
        unique_labels = np.unique(class_labels)
        for class_label in unique_labels:
            df_for_percentage_threshold_by_class = bit_trainings_matrix[np.array(class_labels) == class_label]

            # get frequency of presence and remove features not fulfilling percentage threshold
            number_of_measurements_by_class = df_for_percentage_threshold_by_class.shape[0]
            high_threshold = int(percentage_threshold * number_of_measurements_by_class)
            counts_by_class = df_for_percentage_threshold_by_class.sum()

            # first check if at least high_threshold many measurements per class support peak presence -> >=high_threshold
            filter_mask_by_class[class_label] = counts_by_class >= high_threshold

        # second: unite mask for all classes ->  or
        # np.logical_or is only applicable on 2 elements - so need to create 2-tuples from it if not already binary
        no_class_labels = len(unique_labels)
        if no_class_labels == 2:
            filter_mask_present_in_at_least_one_class = np.logical_or(*filter_mask_by_class.values())
        elif no_class_labels == 3:
            filter_masks = list(filter_mask_by_class.values())
            filter_mask_present_in_at_least_one_class = np.logical_or(np.logical_or(filter_masks[0],filter_masks[1]),
                                                                      filter_masks[2])
        elif no_class_labels == 4:
            filter_masks = list(filter_mask_by_class.values())
            filter_mask_present_in_at_least_one_class = np.logical_or(np.logical_or(filter_masks[0], filter_masks[1]),
                                                                      np.logical_or(filter_masks[2], filter_masks[3]))
        elif no_class_labels == 5:
            filter_masks = list(filter_mask_by_class.values())
            filter_mask_present_in_at_least_one_class = np.logical_or(np.logical_or(filter_masks[0], filter_masks[1]),
                                                        np.logical_or(np.logical_or(filter_masks[2], filter_masks[3]),
                                                                      filter_masks[4]))
        elif no_class_labels == 6:
            filter_masks = list(filter_mask_by_class.values())
            filter_mask_present_in_at_least_one_class = np.logical_or(np.logical_or(filter_masks[0], filter_masks[1]),
                                                        np.logical_or(np.logical_or(filter_masks[2], filter_masks[3]),
                                                                      np.logical_or(filter_masks[4], filter_masks[5])))
        elif no_class_labels == 7:
            filter_masks = list(filter_mask_by_class.values())
            filter_mask_present_in_at_least_one_class = np.logical_or(np.logical_or(filter_masks[0], filter_masks[1]),
                                                        np.logical_or(np.logical_or(filter_masks[2], filter_masks[3]),
                                                        np.logical_or(np.logical_or(filter_masks[4], filter_masks[5]),
                                                                      filter_masks[6])))
        elif no_class_labels == 8:
            filter_masks = list(filter_mask_by_class.values())
            filter_mask_present_in_at_least_one_class = np.logical_or(np.logical_or(filter_masks[0], filter_masks[1]),
                                                        np.logical_or(np.logical_or(filter_masks[2], filter_masks[3]),
                                                        np.logical_or(np.logical_or(filter_masks[4], filter_masks[5]),
                                                                      np.logical_or(filter_masks[6],filter_masks[7]))))
        else:
            raise ValueError(f"Max 8 classes at once supported. Got {no_class_labels}")

        # third: remove features that are present in all classes -> count = 0 or count = number_of_measurement -> false
        counts = bit_trainings_matrix.sum()
        number_of_measurements = bit_trainings_matrix.shape[0]
        # this part used to remove all signals for visualnow - we want to keep positive signals
        # informative_feature_filter_mask = np.logical_not(np.logical_or(counts == number_of_measurements, counts == 0))
        # only remove features that don't appear
        informative_feature_filter_mask = np.logical_not(counts == 0)

        # features need to be present in at least the percentage of classes, and be informative
        return np.logical_and(filter_mask_present_in_at_least_one_class, informative_feature_filter_mask)


    def remove_redundant_features(self, peak_alignment_result, class_label_dict, noise_threshold=0.0001, percentage_threshold=0.5):
        """
        need to remove features which are not present in any measurement -
        to call the presence of a feature we need to apply a threshold - the noise_threshold, everything below will
            be considered noise
        1 determine noise threshold - call presence if satisfies
          we hardcoded a minimum intensity value - which is a factor of maximum intensity
        --> needs to be greater than 10000th of maximum intensity
        Note: intensity values for PEAX are not normalized, so min intensity threshold filtering doesnt apply
        2 distinguish between BitPeakAlignmentResult and FloatPeakAlignmentResult
          Bit PeakResult can immediately be filtered by percentage_threshold
          FloatPeakAlignmentResult needs to be handled by converting to intermediate BitPeak result and then applying percentage_threshold
        :param peak_alignment_result:
        :param class_label_dict:
        :param noise_threshold:
        :param percentage_threshold:
        :return:
        """
        reduced_df_dict = dict()

        if isinstance(peak_alignment_result, FloatPeakAlignmentResult):
            dict_of_df_for_percentage_threshold = dict()
            for peak_detection_method, df_for_prep in peak_alignment_result.dict_of_df.items():

                dict_of_df_for_percentage_threshold[peak_detection_method] = df_for_prep > noise_threshold

        elif isinstance(peak_alignment_result, BitPeakAlignmentResult):
            dict_of_df_for_percentage_threshold = peak_alignment_result.dict_of_df.copy()

        else:
            raise ValueError(
                f"Error while trying to reduce features. " +
                f"Got {type(peak_alignment_result)}, expected FloatPeakAlignmentResult or BitPeakAlignmentResult")

        # we apply the mask to each dataframe in the peakAlignmentResult
        dict_of_filter_masks = dict()

        # get class_labels from description of peak_alignment results
        # class_labels = peak_alignment_result.peak_descriptions['class_label'].values

        for peak_detection_method, df_for_percentage_threshold_all_classes in dict_of_df_for_percentage_threshold.items():
            if not (len(self.class_label_dict)) == df_for_percentage_threshold_all_classes.shape[0]:
                raise ValueError(f"Wrong number of samples in class_labels and trainingsmatrix. {len(self.class_label_dict)} != {len(df_for_percentage_threshold_all_classes.index.values)}")

            # assumption: same order of self.measurements / trainingsmatrix
            if not np.all(np.array(list(self.class_label_dict.keys())) == df_for_percentage_threshold_all_classes.index.values):
                print("TRAININGSMATRIX DEBUG")
                print(f"{np.array(list(self.class_label_dict.keys()))}")
                print(f"{df_for_percentage_threshold_all_classes.index.values}")

                # TODO maybe just sort on the fly if length match? - probably has consequences for other fields though...
                raise ValueError("Peak descriptions order does not match the one in trainingsmatrix")

            # features need to be present in at least the percentage of classes, and be informative
            dict_of_filter_masks[peak_detection_method] = MccImsAnalysis.remove_redundant_features_helper(
                    df_for_percentage_threshold_all_classes, list(self.class_label_dict.values()), percentage_threshold)

        for peak_detection_method, _ in dict_of_df_for_percentage_threshold.items():
            # apply mask to remove all redundant features
            mask = dict_of_filter_masks[peak_detection_method]
            original_matrix = peak_alignment_result.dict_of_df[peak_detection_method]
            number_of_orig_features_above_noise = sum(np.sum(original_matrix > noise_threshold))
            print(f"Reduced {peak_detection_method} AlignmentResult from "
                  f"{number_of_orig_features_above_noise} to {np.sum(mask)} features "
                  f"by applying noise_threshold {noise_threshold} and percentage_threshold {percentage_threshold}")
            # if not np.sum(mask):
            #     # empty mask = removed all features - will lead to error

            # somehow we got NaNs in rows that contained 0.0
            reduced_df_dict[peak_detection_method] = original_matrix[original_matrix.columns[mask]].fillna(0)

        return reduced_df_dict


    def reduce_features(self, feature_reduction_methods, param_map={}):
        """
        Reduce number of features by removing features from trainingsmatrix which do not contribute information
        Only remove negative features, or if not present in any, as noise_threshold can't be properly trusted
        :param feature_reduction_methods:
        :param param_map:
        :return:
        """
        # alternative: sklearn based approach - http://scikit-learn.org/stable/modules/feature_selection.html
        feature_reduction_map = {
            FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES: self.remove_redundant_features,
            # FeatureReductionMethod.REMOVE_CONTROL_PEAKS_POSITIVE: self.remove_control_peaks_positive,  # remove vs control class, only keep intensities higher than control
            # FeatureReductionMethod.REMOVE_CONTROL_PEAKS_NEGATIVE: self.remove_control_peaks_negative,  # remove vs control class, only keep intensities lower than control
            # FeatureReductionMethod.REMOVE_PRE_RIP_PEAKS: self.remove_pre_rip_peaks,  # would need peak coords to operate, as grid can vary
        }

        # if not specified - use the ones given when creating the analysis
        if not param_map:
            param_map = self.performance_measure_parameter_dict

        print(f"Applying feature reduction with {param_map}")
        #TODO we actually need to go through them in specific order

        if FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES in feature_reduction_methods:
            self.peak_alignment_result.dict_of_df = self.remove_redundant_features(
                class_label_dict=self.class_label_dict,
                peak_alignment_result=self.peak_alignment_result, **param_map.get(FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES, {}))

        # TODO apply remove_control_peaks_positive



    def remove_redundant_features_fm(self):
        """
        Takes Feature matrix from analysis result and parameters from evaluation params to call feature reduction
            Updates Analysis result for usage with evaluate performance and further downstream pipeline
        :return:
        """
        reduction_params = self.performance_measure_parameter_dict.get(
                FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES,
                self.DEFAULT_FEATURE_REDUCTION_PARAMETERS[FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES])

        print("Applying feature reduction with {}".format(reduction_params))
        noise_threshold = reduction_params['noise_threshold']
        percentage_threshold = reduction_params['percentage_threshold']

        # TODO also add other feature reduction methods here
        self.analysis_result.trainings_matrix = AnalysisResult.remove_redundant_features_fm(
            self.analysis_result.trainings_matrix, self.analysis_result.class_label_dict,
            noise_threshold=noise_threshold, percentage_threshold=percentage_threshold,
        )

    # TODO implement method for fm based approach only - apply after redundant method
    # @staticmethod
    # def remove_control_peaks_positive():
    #
    #
    # @staticmethod
    # def _remove_control_peaks_positive_helper(matrix, class_label_dict, control_label, quantile_threshold=0.9):
    #     print(
    #         f"Applying POSITIVE_CONTROL_FILTER with control_label={control_label} and quantile_threshold={quantile_threshold}")
    #     reverse_class_labels = defaultdict(list)
    #     for k, v in class_label_dict.items():
    #         reverse_class_labels[v].append(k)
    #
    #     control_samples = []
    #     non_control_samples = []
    #     for k, v in reverse_class_labels.items():
    #         if k == control_label:
    #             control_samples.extend(v)
    #         else:
    #             non_control_samples.extend(v)
    #
    #     # get peaks that are in 90 percent of controls higher than in rest
    #     control_samples = reverse_class_labels[control_label]
    #     control_matrix = matrix.loc[matrix.index.intersection(control_samples)]
    #     control_quantiles = control_matrix.quantile(q=quantile_threshold)
    #     #     print(control_quantiles)
    #
    #     # create mask with quantile thresholds from control samples
    #     non_control_matrix = matrix.loc[matrix.index.intersection(non_control_samples)]
    #     quantile_mask = non_control_matrix < control_quantiles
    #
    #     # needs to apply to all samples in non-control
    #     counts_by_peak = quantile_mask.sum()
    #     no_non_controls = len(non_control_samples)
    #
    #     # remove all peaks that don't fullfill criterium
    #     counts_mask = counts_by_peak < no_non_controls
    #     wanted_columns = matrix.columns[counts_mask].tolist()
    #     result_matrix = matrix.loc[matrix.index, matrix.columns.intersection(wanted_columns)]
    #     print(f"Reduced matrix from {matrix.shape[1]} cols to {result_matrix.shape[1]}")
    #     return result_matrix


    def evaluate_performance(self, param_map={}, exising_analysis_result=None):
        """
        Evaluate performance of models by building random forest in cross validation and compute p_values
        :param param_map:
        :param exising_analysis_result:
        :return:
        """
        # first filter performance_measure_parameter_dict by performance measures
        performance_measure_steps = []
        for pm in self.performance_measure_parameter_dict.keys():
            try:
                performance_measure_steps.append(PerformanceMeasure(pm))
            except ValueError:
                pass
        if not performance_measure_steps:
            raise ValueError("No performance measures specified")
        # use parameter map from setup
        if not param_map:
            param_map = self.performance_measure_parameter_dict

        using_custom_feature_matrix = False
        if (exising_analysis_result is not None) and isinstance(exising_analysis_result, AnalysisResult):
            self.analysis_result = exising_analysis_result
            using_custom_feature_matrix = True

        else:
            self.analysis_result = AnalysisResult(self.peak_alignment_result, self.measurements,
                                              peak_alignment_step=self.peak_alignment_step,
                                              peak_detection_steps=self.peak_detection_combined,
                                              dataset_name=self.dataset_name,
                                              class_label_dict=self.get_class_label_dict()
                                              )

        if PerformanceMeasure.PCA_LDA_CLASSIFICATION in self.performance_measure_parameter_dict:
            raise NotImplementedError("Performance measure {} not yet supported.".format(PerformanceMeasure.PCA_LDA_CLASSIFICATION.name))

        def decision_tree_mock_function(**kwargs):
            """
            Doesn't do anything, just accepts keyword arguments
            :param kwargs:
            :return:
            """
            pass

        performance_measure_map = {
            PerformanceMeasure.FDR_CORRECTED_P_VALUE: self.analysis_result.evaluate_fdr_corrected_p_value,
            # PerformanceMeasure.PCA_LDA_CLASSIFICATION: self.analysis_result.,
            PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION: self.analysis_result.evaluate_random_forest_classifier_roc,
            PerformanceMeasure.DECISION_TREE_TRAINING: decision_tree_mock_function,
        }

        for pm in performance_measure_steps:
            print(f"Applying performance measure {pm} with {param_map.get(pm)}")
            performance_measure_map[pm](**param_map.get(pm, {}))


        # save feature names - makes sure they are json serizable
        feature_names_by_pdm = dict()
        if not using_custom_feature_matrix:
            for peak_detection_method, trainings_matrix in self.peak_alignment_result.dict_of_df.items():
                if isinstance(peak_detection_method, str):
                    peak_detection_method_name = peak_detection_method
                else:
                    peak_detection_method_name = peak_detection_method.name
                feature_names_by_pdm[peak_detection_method_name] = trainings_matrix.columns.values.tolist()
        else:
            for peak_detection_method, trainings_matrix in self.analysis_result.trainings_matrix.items():
                if isinstance(peak_detection_method, str):
                    peak_detection_method_name = peak_detection_method
                else:
                    peak_detection_method_name = peak_detection_method.name
                feature_names_by_pdm[peak_detection_method_name] = trainings_matrix.columns.values.tolist()

        # set the attribute in analysis result
        self.analysis_result.feature_names_by_pdm = feature_names_by_pdm

        # get params for decision tree training
        params_dt = self.performance_measure_parameter_dict.get(PerformanceMeasure.DECISION_TREE_TRAINING, {})
        # crete decision trees for all eval methods and best features
        self.analysis_result.create_decision_trees(**params_dt)


    def get_stats_for_models(self):
        """
        Helper method for `get_best_model()` method - returning eval_df
        :return:
        """
        best_features_df = self.analysis_result.best_features_df
        # best_features = best_features_df.loc[best_features_df['performance_measure_name'] == PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name]

        # get mean roc
        # get mean p_values
        stats_holder_dict = self.analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection
        eval_dicts = []
        # did_perform_cross_val = True
        for pdm in self.peak_detection_combined:
            pdm_name = pdm.name
            stats_dict = stats_holder_dict[PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name][pdm_name]
            if 'error' in stats_dict.keys():
                macro_roc_auc = 0
                pass
            else:
                macro_roc_auc = stats_dict['area_under_curve_macro']
            # merge with p_values from method
            # get mean_p_values
            mean_p_val = np.mean(
                best_features_df[best_features_df['peak_detection_method_name'] == pdm_name]['corrected_p_values'])
            eval_dicts.append({
                'macro_roc_auc': macro_roc_auc,
                'mean_p_val': mean_p_val,
                'peak_detection_method_name': pdm_name,
            })

        eval_df = pd.DataFrame(eval_dicts)
        #  why is eval_df empty? - peak_detection_combined never set - need to add when parsing feature matrix
        eval_df['1-mean_p_val'] = 1 - eval_df['mean_p_val']
        return eval_df


    def get_best_model(self):
        """
        Return the `PeakDetectionMethod`, `PredictionModel` and FeatureNames which produced
        - 1 highest AU_ROC when evaluating the RandomForestClassifiers
        - 2 lowest mean_p_value on selected features
        This model is a RandomForestClassifier
        :return: (best_pdm_name, feature_names, predictor_pickle)
        """
        # model evaluation (ROC) is stored in self.analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection
        # best features /
        # analysis_result.best_features_df.loc[mcc_ims_analysis.analysis_result.best_features_df['performance_measure_name'] == eval_method_name]
        # also provide the random forest model trained on the selected features?

        eval_df = self.get_stats_for_models()

        # get best model row - sorting by roc_auc and p_val
        best_row = eval_df.sort_values(by=['macro_roc_auc', '1-mean_p_val'], ascending=False, )[:1]
        best_method_name = best_row['peak_detection_method_name'].values[0]
        print(f"Selected best model {best_method_name} with auc = {best_row['macro_roc_auc'].values[0]} and mean_p_val = {best_row['mean_p_val'].values[0]}")

        # select best model - sort
        # we can get predictor from self.analysis_result.model_by_pdm
        predictor = self.analysis_result.export_prediction_model(path_to_save="", peak_detection_method_name=best_method_name, use_buffer=True)
        self.best_selected_method_name = best_method_name
        return best_method_name, self.analysis_result.feature_names_by_pdm[best_method_name], predictor


    def export_peak_lists_to_csv(self, directory, use_buffer=False):
        if self.peak_alignment_result.peak_coordinates:
            out_list = []
            output = None
            for peak_detection_method, peak_coords_df in self.peak_alignment_result.peak_coordinates.items():
                if use_buffer:
                    buffer = StringIO()
                    peak_coords_df.to_csv(buffer, sep='\t', float_format='%.6f', index=True, header=True,
                                          index_label="index")
                    buffer.seek(0)
                    output = buffer.getvalue()
                    buffer.close()
                    out_list.append(output)
                else:
                    result_data_dir = "{}results/data/{}/".format(self.dir_level, self.dataset_name)
                    Path(result_data_dir).mkdir(parents=True, exist_ok=True)
                    fn = "{}Peak_list_{}.csv".format(result_data_dir, peak_detection_method)
                    peak_coords_df.to_csv(fn, sep='\t', float_format='%.6f', index=True, header=True,
                                          index_label="index")
            return output

        else:
            raise ExportError("Cannot export peak lists. No peak_coordinates found in the peak_alignment_result.")


    def export_results_to_csv(self, directory):
        """
        Export peak detection results and preprocessed files to csv files
        :param directory:
        :return:
        """
        self.export_preprocessed_measurements(directory)
        # save prediction method in filename, so we can can have multiple runs of peak detection
        # overwriting previous results
        for peak_detection, peak_detection_result_lis in self.peak_detection_results.items():
            for pdr in peak_detection_result_lis: pdr.export_as_csv(directory)


    def import_results_from_csv_list(self, list_of_files, peak_detection_step, based_on_measurement=None):
        """
        Import Peak detection results from csv files, `list_of_files` will be sorted again
        :param list_of_files:
        :param peak_detection_step:
        :param based_on_measurement:
        :return:
        """
        if isinstance(list_of_files, list):
            results = []
            # is list_of_files sorted? otherwise it would be a problem for the feature matrix - as the class labels are sorted
            # went unnoticed for a long time due to adding files to the zips for the server in alphabetical order, not the case locally
            # measurement names will be taken from class_labels
            # this bug took 4 days to find
            list_of_files = sorted(list_of_files)
            for ftm in list_of_files:
                peak_df = PeakDetectionResult.import_from_csv(ftm)
                results.append(
                    PeakDetectionResult(
                        based_on_measurement=based_on_measurement,
                        peak_df=peak_df, import_from_csv=True,
                        peak_detection_method=peak_detection_step))

            self.peak_detection_results[peak_detection_step.name] = results

        else:
            raise TypeError("Expected a list of filenames, got {} instead.".format(type(list_of_files)))


    def import_results_from_csv_dir(self, directory, class_label_file):
        """
        Import peak detection results and preprocessed measurements from csv files found in directory -
            matching peak detection methods with file suffixes
        :param directory:
        :return:
        """
        self.class_label_file = class_label_file

        # import all preprocessed measurements
        measurements_glob_str = "{}*{}.csv".format(directory, "_preprocessed")
        # do a sanity check, whether we have all required measurements in dir, and are not missing any
        print("Starting import process")
        preprocessed_fns = glob.glob(measurements_glob_str)
        self.measurements = self.read_preprocessed_measurements(preprocessed_fns)
        self.assign_class_labels()

        # list all files that have a "_peak_detection_result.csv" file_ending
        peak_detection_results_glob_str = "{}*{}.csv".format(directory,
                                                             PeakDetectionResult.peak_detection_result_suffix)
        files_that_match = glob.glob(peak_detection_results_glob_str)

        result_path_dict = dict()

        pdm_tuples = [(opt.name, opt) for opt in (
                PeakDetectionMethod.JIBB, PeakDetectionMethod.WATERSHED, PeakDetectionMethod.TOPHAT,
                ExternalPeakDetectionMethod.PEAX, PeakDetectionMethod.VISUALNOWLAYER)]

        # separate list of files for each Peak Detection method
        for pdm_str, pdm in pdm_tuples:
            pattern_matching_pdm = "{}{}.csv".format(pdm_str, PeakDetectionResult.peak_detection_result_suffix)
            files_that_match_pdm = [f for f in files_that_match if f.endswith(pattern_matching_pdm)]
            if files_that_match_pdm:
                result_path_dict[pdm_str] = (files_that_match_pdm, pdm)
            # print("{}\nMatching files: {}".format(glob_str, files_that_match_pdm))

        # Sanity check if we have the same number of detection results and preprocessed measurements
        number_of_files_per_method = [
            (len(files_that_match_pdm), pdm) for files_that_match_pdm, pdm in result_path_dict.values()
                                     ]
        # it is not required to have preprocessed measurements for all pdr, only nice to have for plots
        print(number_of_files_per_method, (len(self.measurements), "Preprocessed"))

        not_import_error = False
        if self.measurements:
            # same number as preprocessed measurements as pdrs
            not_import_error = all([len(self.measurements) == nfpm for nfpm, pdm in number_of_files_per_method])
            if not_import_error:
                for pdm_str, (files_that_match_pdm, pdm) in result_path_dict.items():
                    self.import_results_from_csv_list(files_that_match_pdm, peak_detection_step=pdm)
        else:
            # same number as of pdrs for all methods
            first_val = number_of_files_per_method[0][0]
            not_import_error = all([first_val == nfpm for nfpm, pdm in number_of_files_per_method])
            if not_import_error:
                for pdm_str, (files_that_match_pdm, pdm) in result_path_dict.items():
                    self.import_results_from_csv_list(files_that_match_pdm, peak_detection_step=pdm)

        # if import test failed
        if not not_import_error:
            # get difference in files - so we can remove them manually
            from collections import defaultdict
            file_counter = defaultdict(list)
            for k, files in result_path_dict.items():
                prefixes = []
                for name in files[0]:
                    bd_pre = name.split("/")[-1]
                    bd_prefix = "_".join(bd_pre.split("_")[:2])
                    # bd_prefixes = ["_".join(name.split("/")[:-1]) for name in files[0]]
                    file_counter[bd_prefix].append(k)

                # file_counter.update(prefixes)
            # print(file_counter)
            out_of_order_prefixes = []
            for k, v in file_counter.items():
                if len(v) != 5:
                    out_of_order_prefixes.append(k)
            # print(out_of_order_prefixes)

            raise ResumeAnalysisError("Import of PeakDetectionResults failed. Number of preprocessed files and peak detection results per method don't match.")

        # do sanity check after importing and raise error if no preliminary results found
        if not any(self.peak_detection_results.values()):
            raise ResumeAnalysisError("Import of PeakDetectionResults failed. No files that match pattern found.")


    def get_peak_id_coord_map(self, threshold_inverse_reduced_mobility=0.015, threshold_scaling_retention_time=0.1):
        """
        Return Probe clustering grid used to determine peak_ids
        :param threshold_inverse_reduced_mobility:
        :param threshold_scaling_retention_time:
        :return:
        """
        if self.peak_id_coord_map is None:
            self.peak_id_coord_map = self._compute_peak_id_coord_map(threshold_inverse_reduced_mobility,
                                                                     threshold_scaling_retention_time)
        return self.peak_id_coord_map


    @staticmethod
    def _compute_peak_id_coord_map(threshold_inverse_reduced_mobility=0.015, threshold_scaling_retention_time=0.1):
        x_bounds, y_bounds, y_centers = MccImsAnalysis.compute_window_size(
            threshold_inverse_reduced_mobility=threshold_inverse_reduced_mobility,
            threshold_scaling_retention_time=threshold_scaling_retention_time,
            return_y_centers=True)

        the_map = {}
        for i, x in enumerate(x_bounds):
            # we dont want an out of bounds error
            x1 = x_bounds[min(len(x_bounds)-1, i+1)]
            for j, y in enumerate(y_bounds):
                y1 = y_bounds[min(len(y_bounds) - 1, j + 1)]
                cluster_no = int(j * len(x_bounds) + i)
                peak_id = "Peak_{0:0>4}".format(cluster_no)
                # bounding box -> allows to compute simple radius by adjusting center
                the_map[peak_id] = {
                    "inverse_reduced_mobility": ((x+x1)/2.),
                    "retention_time": ((y+y1)/2.),
                    "radius_inverse_reduced_mobility": ((x1-x)/2.),
                    "radius_retention_time": ((y1-y)/2.),
                    "cluster_no": cluster_no,
                                    }
        return the_map

    # Available preprocessing options
    # tuples are basically frozenlists - so immutable
    AVAILABLE_NORMALIZATION_METHODS = (
        NormalizationMethod.BASELINE_CORRECTION,
        NormalizationMethod.INTENSITY_NORMALIZATION
    )

    AVAILABLE_PEAK_DETECTION_METHODS = (
        PeakDetectionMethod.TOPHAT,
        PeakDetectionMethod.WATERSHED,
        PeakDetectionMethod.JIBB,
        PeakDetectionMethod.VISUALNOWLAYER
    )

    AVAILABLE_DENOISING_METHODS = (
        DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY,
        DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION,
        DenoisingMethod.GAUSSIAN_FILTER,
        DenoisingMethod.MEDIAN_FILTER,
        DenoisingMethod.NOISE_SUBTRACTION,
        DenoisingMethod.SAVITZKY_GOLAY_FILTER)

    AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS = (
        ExternalPeakDetectionMethod.PEAX,
        ExternalPeakDetectionMethod.CUSTOM,)

    AVAILABLE_PEAK_ALIGNMENT_METHODS = (
        # PeakAlignmentMethod.WINDOW_FRAME,
        # PeakAlignmentMethod.WARD_CLUSTERING,
        PeakAlignmentMethod.PROBE_CLUSTERING,
        # PeakAlignmentMethod.MEAN_SHIFT_CLUSTERING,
        # PeakAlignmentMethod.K_MEANS_CLUSTERING,
        PeakAlignmentMethod.DB_SCAN_CLUSTERING,
        # PeakAlignmentMethod.AFFINITY_PROPAGATION_CLUSTERING
    )

    AVAILABLE_FEATURE_REDUCTION_METHODS = (
        FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES,
    )

    DEFAULT_FEATURE_REDUCTION_PARAMETERS = {
        FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES: {
            "noise_threshold": 0.0001,  # dangerous option - can easily lead to loss of all signals
            "percentage_threshold": 0.5,  # also dangerous - need to have signal in x% of masurements of one class
        },
    }

    AVAILABLE_PERFORMANCE_MEASURES = (
        PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION,
        # PerformanceMeasure.PCA_LDA_CLASSIFICATION,
        PerformanceMeasure.FDR_CORRECTED_P_VALUE,
        PerformanceMeasure.DECISION_TREE_TRAINING,
    )

    DEFAULT_PERFORMANCE_MEASURE_PARAMETERS = {
        PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION: {
            'n_splits_cross_validation': 3,
            'n_estimators_random_forest': 2000,
            'n_of_features': 10,  # Best feature selection
        },
        PerformanceMeasure.FDR_CORRECTED_P_VALUE: {
            'benjamini_hochberg_alpha': 0.05,
            'n_of_features': 10,
            # 'n_permutations': 1000,
            #     TODO add remove_insignificant_cutoff - only keeping p_values smaller than 0.01 after correction?
        },
        PerformanceMeasure.DECISION_TREE_TRAINING: {
            'max_depth': 5,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            # 'max_features': 10,  # number of features considered when splitting a node, not in total
        },
        PerformanceMeasure.PCA_LDA_CLASSIFICATION: {
        },
    }

    AVAILABLE_EVALUATION_MEASURES = tuple(chain(
        AVAILABLE_PERFORMANCE_MEASURES,
        AVAILABLE_FEATURE_REDUCTION_METHODS, ))

    DEFAULT_EVALUATION_MEASURE_PARAMETERS = DEFAULT_PERFORMANCE_MEASURE_PARAMETERS.copy()
    DEFAULT_EVALUATION_MEASURE_PARAMETERS.update(DEFAULT_FEATURE_REDUCTION_PARAMETERS)

    AVAILABLE_PREPROCESSING_METHODS = tuple(chain(
        AVAILABLE_NORMALIZATION_METHODS,
        AVAILABLE_DENOISING_METHODS,
        AVAILABLE_PEAK_DETECTION_METHODS,
        AVAILABLE_PEAK_ALIGNMENT_METHODS,
        AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS))

    BASIC_THRESHOLD_INVERSE_REDUCED_MOBILITY = 0.015
    BASIC_THRESHOLD_SCALING_RETENTION_TIME = 0.1

    DEFAULT_PREPROCESSING_PARAMETERS = {
        NormalizationMethod.BASELINE_CORRECTION: {},
        NormalizationMethod.INTENSITY_NORMALIZATION: {},
        DenoisingMethod.GAUSSIAN_FILTER: {'sigma': 1},
        DenoisingMethod.SAVITZKY_GOLAY_FILTER: {'window_length': 9,
                                                'poly_order': 2},
        DenoisingMethod.NOISE_SUBTRACTION: {'cutoff_1ko_axis': 0.4},
        DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY: {'cutoff_1ko_axis': 0.4},
        DenoisingMethod.MEDIAN_FILTER: {'kernel_size': 9},
        DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION: {'level_inverse_reduced_mobility': 4,
                                                          'level_retention_time': 2},
        PeakDetectionMethod.WATERSHED: {'noise_threshold': 1.5, },
        PeakDetectionMethod.TOPHAT: {'noise_threshold': 1.4, },
        PeakDetectionMethod.JIBB: {'noise_threshold': 1.5,
                                   'range_ivr': 5,
                                   'range_rt': 7},
        PeakDetectionMethod.VISUALNOWLAYER: {  # no default for visualnow - requires additional file
        },
        ExternalPeakDetectionMethod.PEAX: {  # no default for PEAX
        },
        ExternalPeakDetectionMethod.CUSTOM: {  # no default for CUSTOM - but needed for parameter matching
        },
        PeakAlignmentMethod.AFFINITY_PROPAGATION_CLUSTERING: {'preference': -0.1},
        PeakAlignmentMethod.DB_SCAN_CLUSTERING: {'eps': 0.025,
                                                 'min_samples': 2},
        PeakAlignmentMethod.K_MEANS_CLUSTERING: {'n_clusters': 80},
        PeakAlignmentMethod.MEAN_SHIFT_CLUSTERING: {'bandwidth': 0.1},
        PeakAlignmentMethod.WINDOW_FRAME: {'distance_threshold': (3, 0.02)},
        PeakAlignmentMethod.PROBE_CLUSTERING: {
            'threshold_inverse_reduced_mobility': BASIC_THRESHOLD_INVERSE_REDUCED_MOBILITY,
            'threshold_scaling_retention_time': BASIC_THRESHOLD_SCALING_RETENTION_TIME
        },
    }


class PeakDetectionResult(object):
    """
    Result of a Peak detection method
    Holds list of peaks with the columns ['measurement_name', 'peak_id', 'retention_time', 'inverse_reduced_mobility', 'intensity']
        where peak_id does not need to be unique - will be consumed by peak-alignment method to form a PeakAlignmentResult
    """
    peak_detection_result_suffix = "_peak_detection_result"
    _peak_detection_original_result_suffix = "_peak_detection_result"

    def __init__(self, based_on_measurement, peak_df, peak_detection_method, import_from_csv=False, is_gcms_based=False):
        """
        :param based_on_measurement:
        :param peak_df:
        :param peak_detection_method:
        :param import_from_csv:
        :param is_gcms_based: is this pdr imported from gcms? if true - "inverse_reduced_mobility" actually holds MZ (mass-charge) value
        """
        self.imported_from_csv = import_from_csv
        # if filename still has csv ending, we should split it out

        if import_from_csv:
            self.measurement_name = peak_df['measurement_name'][0]
        else:
            # fill the column measurement_name with the measurements name
            self.measurement_name = based_on_measurement.filename
            peak_df['measurement_name'] = [based_on_measurement.filename] * peak_df.shape[0]
        self.peak_df = peak_df

        self.peak_detection_step = peak_detection_method
        self.is_gcms_based = is_gcms_based


    def __repr__(self):
        str_imported = ""
        if self.imported_from_csv:
            str_imported = ", imported from csv"
        return "PeakDetectionResult ({}, {}{})".format(self.measurement_name, self.peak_detection_step.name, str_imported)


    def export_as_csv(self, directory, use_buffer=False):
        """
        Save peak detection matrix as csv file
        :return:
        """
        peak_detection_method = self.peak_detection_step

        if self.measurement_name.endswith(PeakDetectionResult.peak_detection_result_suffix):
            outname = self.measurement_name.rsplit(PeakDetectionResult.peak_detection_result_suffix)[0]
        else:
            outname = self.measurement_name

        filename = "{}{}_{}{}.csv".format(directory, outname, peak_detection_method.name, PeakDetectionResult.peak_detection_result_suffix)
        columns = ['measurement_name',
                   'peak_id',
                   'retention_time',
                   'inverse_reduced_mobility',
                   'intensity']
        rv = None
        if use_buffer:
            buffer = StringIO()
            self.peak_df.to_csv(buffer, sep='\t', float_format='%.5f', index=False, header=True, columns=columns)
            buffer.seek(0)
            rv = buffer.getvalue()
            buffer.close()

        else:
            self.peak_df.to_csv(filename, sep='\t', float_format='%.5f', index=False, header=True, columns=columns)

        return rv


    @staticmethod
    def import_from_csv(inname):
        """
        Read annotated peaks from csv file and load into Dataframe
        :return: pd.DataFrame
        """
        peak_df = pd.read_csv(inname, sep="\t", comment="#")
        # check whether column names are as expected
        expected_columns = {'measurement_name', 'peak_id', 'retention_time', 'inverse_reduced_mobility', 'intensity'}
        column_vals_from_peax = {'measurement_name', 'peak_name', 't', 'r', 'signal', 'index_t', 'index_r'}
        if set(peak_df.columns) == expected_columns:
            peak_df.intensity.fillna(0.0, inplace=True)

            if peak_df.peak_id.dtype == np.float64:
                peak_df.peak_id = np.array(peak_df.peak_id, dtype=int)


        elif set(peak_df.columns) == column_vals_from_peax:
            # need to remap column names
            new_df = peak_df.assign(peak_id=peak_df.peak_name, retention_time=peak_df.r, inverse_reduced_mobility=peak_df.t, intensity=peak_df.signal)
            # drop all columns that are not needed
            new_df.intensity.fillna(0.0, inplace=True)
            peak_df = new_df.drop(['peak_name', 'r', 't', 'signal', 'index_r', 'index_t'], axis=1)
        else:
            assert ValueError(f'Read in column names do not match. Got {expected_columns}, expected {peak_df.columns}')
        return peak_df.sort_values(by=['retention_time', 'inverse_reduced_mobility']).reindex()


class AnalysisResult(object):
    """
    The Result of an Analysis.
    Holds class labels, training matrix and statistics
    """
    def __init__(self, peak_alignment_result, based_on_measurements, peak_detection_steps, peak_alignment_step, dataset_name, class_label_dict={}, feature_matrix=None):
        """
        If feature_matrix is not none - initialize from it instead of peak_alignment_result
        :param peak_alignment_result:
        :param based_on_measurements:
        :param performance_measure:
        :param peak_detection_steps: List of peak detection steps - instances of PeakDetectionMethod and ExternalPeakDetectionMethod
        :param peak_alignment_step:
        :param dataset_name:
        :param n_of_reported_peaks:
        """
        self.feature_matrix = feature_matrix
        if self.feature_matrix is not None:
            self.has_peak_alignment_result = False
            self.trainings_matrix = feature_matrix
            self.peak_coordinates = {pds.name:None for pds in peak_detection_steps}

        else:
            self.peak_alignment_result = peak_alignment_result
            self.has_peak_alignment_result = True
            self.peak_coordinates = peak_alignment_result.peak_coordinates
            self.trainings_matrix = peak_alignment_result.dict_of_df


        self.measurements = based_on_measurements

        if class_label_dict:
            self.class_label_dict = class_label_dict
            self.class_labels = list(class_label_dict.values())
        else:
            self.class_label_dict = OrderedDict({m.filename: m.class_label for m in based_on_measurements})
            self.class_labels = [m.class_label for m in based_on_measurements]


        self.peak_detection_steps = peak_detection_steps
        self.peak_alignment_step = peak_alignment_step
        self.dataset_name = dataset_name
        # self.analysis_statistics_per_peak_detection = dict()
        self.analysis_statistics_per_evaluation_method_and_peak_detection = dict()
        self._pvalues_df = None
        self.peak_intensities_by_pdm_by_class_by_peak = dict()
        self.decision_tree_per_evaluation_method_and_peak_detection = dict() # holds decision tree dot data
        self.decision_tree_model_per_evaluation_method_and_peak_detection = dict() # holds pickles of decision tree
        self.model_by_pdm = dict()
        self.feature_names_by_pdm = dict()
        # self.pvalues_df

        # field holding dataframes of best features to simplify export / import
        self.best_features_df = None


    def __repr__(self):
        num_measurements = self.trainings_matrix[self.peak_detection_steps[0].name].shape[0]
        return f"AnalysisResult of dataset '{self.dataset_name}' with {self.peak_detection_steps} {self.peak_alignment_step} for {num_measurements} measurements"


    @staticmethod
    def remove_redundant_features_fm(feature_matrix, class_label_dict, noise_threshold=0.0001,
                                  percentage_threshold=0.5):
        """
        need to remove features which are not present in any measurement -
        to call the presence of a feature we need to apply a threshold - the noise_threshold, everything below will
        be considered noise
        1 determine noise threshold - call presence if satisfies
          we hardcoded a minimum intensity value - which is a factor of maximum intensity
        --> needs to be greater than 10000th of maximum intensity
        2 FeatureMatrix needs to be handled by converting to intermediate BitPeak result and then applying percentage_threshold

        :param feature_matrix:
        :param class_label_dict:
        :param noise_threshold:
        :param percentage_threshold:
        :return:
        """
        reduced_df_dict = dict()
        # "do z-normalization" -> convert to boolean matrix -> apply filter -> "revert z-normalization"
        # not actually applying z-normalization

        dict_of_df_for_percentage_threshold = dict()
        for peak_detection_method, df_for_prep in feature_matrix.items():

            # make sure order matches
            sample_names = np.array(list(class_label_dict.keys()))
            if not np.all(sample_names == df_for_prep.index.values):
                raise ValueError("Measurement order between class labels and trainingsmatrix does not match!")
            max_val = max(np.max(np.abs(df_for_prep)))
            dict_of_df_for_percentage_threshold[peak_detection_method] = (df_for_prep / max_val) > noise_threshold


        # we apply the mask to each dataframe in the feature matrix
        dict_of_filter_masks = dict()

        for peak_detection_method, df_for_percentage_threshold_all_classes in dict_of_df_for_percentage_threshold.items():

            # features need to be present in at least the percentage of classes, and be informative
            dict_of_filter_masks[peak_detection_method] = MccImsAnalysis.remove_redundant_features_helper(
                df_for_percentage_threshold_all_classes, list(class_label_dict.values()), percentage_threshold)

        # apply mask
        for peak_detection_method, _ in dict_of_df_for_percentage_threshold.items():
            # apply mask to remove all redundant features
            mask = dict_of_filter_masks[peak_detection_method]
            original_matrix = feature_matrix[peak_detection_method]
            print(
                "Reduced {} FeatureMatrix to {} features by applying noise_threshold {} and percentage_threshold {}".format(
                    peak_detection_method, np.sum(mask), noise_threshold, percentage_threshold))
            # somehow we got NaNs in rows that contained 0.0
            reduced_df_dict[peak_detection_method] = original_matrix[original_matrix.columns[mask]].fillna(0)
        return reduced_df_dict


    def get_pvalues_df(self, n_features=-1, benjamini_hochberg_alpha=0.05):
        """
        Get FDR corrected p_values. Limit reported values to `n_features`
        :param n_features: int
        :param benjamini_hochberg_alpha: float
        :return:
        """
        if self._pvalues_df is None:
            if self.has_peak_alignment_result:
                self._pvalues_df = self._compute_p_values(benjamini_hochberg_alpha=benjamini_hochberg_alpha)
            else:
                self._pvalues_df = self._compute_p_values_custom(benjamini_hochberg_alpha=benjamini_hochberg_alpha)

        if n_features == -1:
            # return all computed p_values
            return self._pvalues_df
        elif not n_features:
            raise ValueError(f"n_features needs to be >0, or for all features -1. Got {n_features}")
        else:
            reduced_df_with_fdr_correction_list = []
            # only one decision tree will be build for each peak detection method
            # to select which peaks to remove, we need to find highest p_values and remove those
            # compute mean over all class comparisons by joining by peak_id
            for pdm_name in self.peak_intensities_by_pdm_by_class_by_peak.keys():
                df_by_pdm_name = self._pvalues_df[
                    self._pvalues_df['peak_detection_method_name'] == pdm_name]
                # get peak ids of lowest peak values to keep
                lowest_pvals_pids = df_by_pdm_name.groupby('peak_id').agg(lambda x: np.mean(x)).nsmallest(n_features,
                                                                                                          'corrected_p_values').index.values
                reduced_df_with_fdr_correction_list.append(
                    df_by_pdm_name.loc[df_by_pdm_name['peak_id'].isin(lowest_pvals_pids)])

            # build reduced dataframe
            return pd.concat(reduced_df_with_fdr_correction_list)


    @staticmethod
    def construct_labels(labels):
        class_labels = np.sort(np.unique(labels))
        is_binary_case = (len(class_labels) == 2)
        # binarize the class labels
        if is_binary_case:
            Y = label_binarize(labels, classes=class_labels)
            Y = Y.ravel()
        else:
            # need multiclass labels, not multilabel classes, see https://scikit-learn.org/stable/modules/multiclass.html
            # issue with labels - use pd.factorize across all label handling
            Y, _ = pd.factorize(labels, sort=True)
            # Y = factor[0]
        return Y


    def _evaluate_fdr_corrected_p_value_helper(self, benjamini_hochberg_alpha, **kwargs):
        """
        Computes corrected fdr without applying a filter or limiting the number of features or permutation test
        :param benjamini_hochberg_alpha:
        :param kwargs:
        :return:
        """

        # use FDR to correct pvalues from MWU
        # get p_value for peaks
        # peak_ids need to be in trainings_matrix - otherwise we don't need to rank them - done in feature reduction

        if self.has_peak_alignment_result:
            feature_set_by_pdm_name = dict()
            for peak_detection_method_name in self.peak_coordinates.keys():
                feature_set_by_pdm_name[peak_detection_method_name] = frozenset(self.trainings_matrix[peak_detection_method_name].columns.values)

            # only works if using FloatPeakAlignmentResults - partly extracting intensities from PeakAlignment /
            #   PeakDetection Results instead of measurements - will extract from measurements unless PEAX
            self.peak_intensities_by_pdm_by_class_by_peak = AnalysisResult.intensities_for_features(self.peak_coordinates, self.measurements, feature_set_by_pdm_name=feature_set_by_pdm_name, trainings_matrix_dict=self.peak_alignment_result.dict_of_df, class_label_dict=self.class_label_dict)

        # we initialize from a feature matrix not a peak-alignment result
        else:
            feature_set_by_pdm_name = dict()

            for peak_detection_method in self.peak_detection_steps:
                peak_detection_method_name = peak_detection_method.name
                feature_set_by_pdm_name[peak_detection_method_name] = frozenset(
                    self.trainings_matrix[peak_detection_method_name].columns.values)
            # print(feature_set_by_pdm_name)

            # only works if using FloatPeakAlignmentResults - extracting intensities from PeakAlignment / PeakDetection Results
            self.peak_intensities_by_pdm_by_class_by_peak = AnalysisResult.intensities_for_features_custom(
                    feature_set_by_pdm_name=feature_set_by_pdm_name,
                    trainings_matrix_dict=self.trainings_matrix,
                    class_label_dict=self.class_label_dict)


        # we dont neccessarily have the same number of samples per class, so a dataframe doesnt make sense - have to save as dict structure

        mwu_new_computation = AnalysisResult.mann_whitney_u_per_dict(self.peak_intensities_by_pdm_by_class_by_peak)

        # we need to correct for fdr before selecting the top n_of_features from mwu_new_computation, otherwise the result will be off
        return AnalysisResult.benjamini_hochberg_for_mann_whitney_u(mwu_new_computation, benjamini_hochberg_alpha=benjamini_hochberg_alpha)


    def _compute_p_values(self, benjamini_hochberg_alpha=0.05):
        # add peak coordinates to resulting dataframe
        return AnalysisResult._add_peak_coordinates_to_df(
                peak_coordinates_dict=self.peak_coordinates,
                df=self._evaluate_fdr_corrected_p_value_helper(benjamini_hochberg_alpha=benjamini_hochberg_alpha), )


    def _compute_p_values_custom(self, benjamini_hochberg_alpha=0.05):
        # DONT add peak coordinates to resulting dataframe - as we don't have coords available
        return self._evaluate_fdr_corrected_p_value_helper(benjamini_hochberg_alpha=benjamini_hochberg_alpha)


    def evaluate_fdr_corrected_p_value(self, n_of_features, benjamini_hochberg_alpha, **kwargs):
        """
        compute p_values with man whitney u test
        correct for FDR with benjamini hochberg
        apply permutation test
        :param n_of_features:
        :param benjamini_hochberg_alpha:
        :param kwargs:
        :return:
        """
        performance_measure = PerformanceMeasure.FDR_CORRECTED_P_VALUE

        df_with_fdr_correction = self.get_pvalues_df(n_features=n_of_features, benjamini_hochberg_alpha=benjamini_hochberg_alpha)
        # permutaion_results = AnalysisResult.permutation_test(self.peak_intensities_by_pdm_by_class_by_peak,
        #                                                      n_permutations=n_permutations)
        # filtered_perm_results = {
        #     pd_: {p_id: val for p_id, val in by_peaks.items() if any([pval <= 0.05 for comp_str, pval in val.items()])}
        #     for pd_, by_peaks in permutaion_results.items()}
        # pvalues -> if both are below 0.05, then we have something
        # assign pval_100fold_permutation_test column

        # df_with_fdr_correction['pval_100fold_permutation_test'] = 'NaN'
        # for pdm, by_peaks in filtered_perm_results.items():
        #     for p_id, class_dict in by_peaks.items():
        #         for comp_str, permut_pval in class_dict.items():
        #             tmp1 = df_with_fdr_correction.peak_id == p_id
        #             tmp2 = df_with_fdr_correction.class_comparison == comp_str
        #             tmp3 = df_with_fdr_correction.peak_detection_method_name == pdm
        #             # df_with_fdr_correction[tmp1 & tmp2 & tmp3]['pval_100fold_permutation_test'] = permut_pval
        #             the_mask = tmp1 & tmp2 & tmp3
        #             df_with_fdr_correction.loc[the_mask, 'pval_100fold_permutation_test'] = permut_pval

        for pdm_name, stats_by_pdm in df_with_fdr_correction.groupby('peak_detection_method_name'):
            for comp_str, stat_by_comp_str in stats_by_pdm.groupby('class_comparison'):
                self.add_to_best_features_df(
                    pdm_name=pdm_name,
                    performance_measure_name=performance_measure.name,
                    best_features=stat_by_comp_str,
                    class_comp_str=comp_str,
                )


    def add_to_best_features_df(self, best_features, pdm_name, performance_measure_name, class_comp_str):
        # add rows for decision tree and statistics table
        # will have a class_str
        assert isinstance(best_features, pd.DataFrame)
        new_df = best_features.copy(deep=True)
        new_df['class_comparison'] = class_comp_str
        new_df['peak_detection_method_name'] = pdm_name
        new_df['performance_measure_name'] = performance_measure_name
        # best_features_with_class_lis.append(new_df)

        if self.best_features_df is None:
            self.best_features_df = pd.DataFrame()
        self.best_features_df = pd.concat([self.best_features_df, new_df], sort=False)


    def evaluate_random_forest_classifier_roc(self, n_estimators_random_forest, n_splits_cross_validation, n_of_features, **kwargs):
        performance_measure = PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION
        self.analysis_statistics_per_evaluation_method_and_peak_detection[performance_measure.name] = {}

        # is it possible to cross validate?
        if self.measurements:
            class_labels = [m.class_label for m in self.measurements]
            class_counts = Counter(class_labels)
            min_occurence = min(class_counts.values())
        else:
            class_labels = list(self.class_label_dict.values())
            class_counts = Counter(class_labels)
            min_occurence = min(class_counts.values())

        can_cross_validate = min_occurence >= 10
        n_splits_cross_validation = min(n_splits_cross_validation, min_occurence)

        for pdm in self.peak_detection_steps:
            if can_cross_validate:
                stats = self._evaluate_random_forest_classifier_roc_helper(
                    trainings_matrix=self.trainings_matrix[pdm.name],
                    peak_detection_name=pdm.name,
                    n_estimators_random_forest=n_estimators_random_forest,
                    n_splits_cross_validation=n_splits_cross_validation,
                    n_of_reported_peaks=n_of_features)
                self.analysis_statistics_per_evaluation_method_and_peak_detection[performance_measure.name][pdm.name] = stats
            else:
                self.analysis_statistics_per_evaluation_method_and_peak_detection[
                    performance_measure.name][pdm.name] = {
                    'error':
                        'CrossValidationError. \n Need more measurements to make splits for cross validation.\n' +
                        'Smalles class has only {} occurences.'.format(min_occurence)}

            # save RandomForest Model pickle in dict
            self.model_by_pdm[pdm] = AnalysisResult.train_random_forest_classifier(
                trainings_matrix=self.trainings_matrix[pdm.name],
                trainings_labels=AnalysisResult.construct_labels(self.class_labels),
                original_labels=self.class_labels,
                n_estimators=n_estimators_random_forest)

            # save feature names for later use
            self.feature_names_by_pdm[pdm.name] = self.trainings_matrix[pdm.name].columns.values

            # compute p_values for best features
            best_features_gini = self.extract_best_features_gini(
                model=self.model_by_pdm[pdm],
                class_labels=class_labels,
                trainings_matrix=self.trainings_matrix[pdm.name],
                peak_coordinates=self.peak_coordinates[pdm.name],
                n_of_reported_peaks=n_of_features,
                dataset_name=self.dataset_name,
                peak_detection_name=pdm.name,
            )

            for class_comp_str, best_features_df in best_features_gini.items():
                self.add_to_best_features_df(pdm_name=pdm.name, performance_measure_name=performance_measure.name,
                                             best_features=best_features_df, class_comp_str=class_comp_str)
            # self.analysis_statistics_per_evaluation_method_and_peak_detection[performance_measure.name][
            #     pdm.name]['best_features'] = best_features_gini
            
            # decision tree vis has been move to view
            # need to save features sorted by gini when not using man whitney
            # self.decision_tree_of_best_features(
            #     trainings_matrix=self.trainings_matrix[pdm.name],
            #     model=self.model_by_pdm[pdm],
            #     peak_detection_name=pdm.name,
            #     evaluation_method_name=performance_measure.name,)


    def _evaluate_random_forest_classifier_roc_helper(self, trainings_matrix, peak_detection_name, n_estimators_random_forest, n_splits_cross_validation, n_of_reported_peaks):
        # workflow from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        # Create the trainings matrix and get the ordered, unique class labels

        X = trainings_matrix.values.astype(float)
        class_labels = np.unique(self.class_labels)
        is_binary_case = (len(class_labels) == 2)

        # the class labels - distinguish between binary and non-bin case
        Y = AnalysisResult.construct_labels(self.class_labels)

        #     # Export model
        # Check, if its supposed to train a new model or to import an existing one
        # Achieve reproducible splits by setting the random_state parameter to a default number
        # only split to estimate error rate with 5 fold cross validation
        # train on whole dataset - careful with much variance
        if is_binary_case:
            mean_std_evaluations = AnalysisResult.cross_validate_binary_model(
                trainings_matrix=X, trainings_labels=Y, original_labels=self.class_labels,
                n_estimators=n_estimators_random_forest, n_splits=n_splits_cross_validation)
        else:
            mean_std_evaluations = AnalysisResult.cross_validate_multiclass_model(
                trainings_matrix=X, trainings_labels=Y, original_labels=self.class_labels,
                n_estimators=n_estimators_random_forest, n_splits=n_splits_cross_validation)

        statistic = dict()
        # statistic['best_features'] = best_features

        statistic["accuracy"], statistic["standard_deviation_accuracy"] = mean_std_evaluations['accuracy']
        statistic["precision"], statistic["standard_deviation_precision"] = mean_std_evaluations['average_precision']

        if is_binary_case:
            # if binary we can simply take the result for the first class, but else we need to apply special formating
            # index_macro = np.argmax(mean_std_evaluations['auc_measures']['mean_auc'])
            statistic["area_under_curve_macro"] = mean_std_evaluations['auc_measures']['mean_auc']
            statistic["standard_deviation_auc_macro"] = mean_std_evaluations['auc_measures']['std_auc']
            statistic["F1-Score"], statistic["standard_deviation_F1-Score"] = mean_std_evaluations['f1']
            statistic["recall"], statistic["standard_deviation_recall"] = mean_std_evaluations['recall']
        else:
            # we do macro averaging in cross_validate_multiclass_model
            statistic["area_under_curve_macro"] = mean_std_evaluations['auc_measures']['mean_auc']
            statistic["standard_deviation_auc_macro"] = mean_std_evaluations['auc_measures']['std_auc']
            statistic["F1-Score"], statistic["standard_deviation_F1-Score"] = mean_std_evaluations['f1']
            statistic["recall"], statistic["standard_deviation_recall"] = mean_std_evaluations['recall']


        statistic['auc_measures'] = mean_std_evaluations['auc_measures']
        def special_formating_for_statistics(stat_dict):
            if isinstance(stat_dict, dict):
                out_str = ""
                for k, v in stat_dict.items():
                    out_str += " {}: {:.5f}".format(k, v)
                return out_str
            else:
                return "{:.5f}".format(stat_dict)

        # need to format the metrics which can have multiple classes - need for reporting
        # compose statistics report and round to 5 digits
        statistics_2nd_str = "\n\nThe statistics of the evaluation are: \n" \
                             "Precision: {:.5f} +- ({:.5f})\n" \
                             "Recall: {} +- ({})\n" \
                             "Roc_AUC: {} +- ({})\n" \
                             "Accuracy: {:.5f} +- ({:.5f})\n" \
                             "F1-Score: {} +- ({})\n".format(
            statistic["precision"], statistic["standard_deviation_precision"],
            statistic["recall"], statistic["standard_deviation_recall"],
            statistic["area_under_curve_macro"],
            special_formating_for_statistics(statistic["standard_deviation_auc_macro"]),
            statistic["accuracy"], statistic["standard_deviation_accuracy"],
            statistic["F1-Score"], statistic["standard_deviation_F1-Score"],
        )

        print("The {}-method was applied as peak detection. {}".format(peak_detection_name, statistics_2nd_str))

        #
        # statistic['intensities_per_class_and_best_feature'], statistic[
        #     'mann_whitney_u_of_best_feature'] = AnalysisResult.intensities_per_class_for_best_features_mann_whitney(
        #     best_features=statistic['best_features'],
        #     measurements=self.measurements)
        #
        # mann_whitney_u_str = "Mann Whitney U Test of best features:\n{}".format(
        #     pformat(statistic['mann_whitney_u_of_best_feature']))

        # corrected_mann_whitney_u_str = "Mann Whitney U Test corrected for FDR using Benjamini Hochberg test:\n{}".format(pformat(self.statistic['mann_whitney_u_of_best_feature']))

        statistic['cross_validation_evaluation_string'] = statistics_2nd_str
        # TODO get intersection between classes using self.statistic['best_features']
        return statistic


    @staticmethod
    def _add_peak_coordinates_to_df(peak_coordinates_dict, df):
        new_df_lis = []
        # or we just merge the dataframes by peakId
        for pdm, coords in peak_coordinates_dict.items():
            df_by_pdm = pd.DataFrame(df[df.peak_detection_method_name == pdm])
            new_df_lis.append(pd.merge(coords, df_by_pdm, on='peak_id'))
        return pd.concat(new_df_lis)


    @staticmethod
    def predict_from_model(model, trainings_matrix):
        return model.predict_proba(trainings_matrix)


    def export_prediction_model(self, path_to_save, peak_detection_method_name, use_buffer=False):
        result = None
        pdm = None
        # convert name to PeakDetectionMethod / ExternalPeakDetectionMethod
        try:
            pdm = PeakDetectionMethod(peak_detection_method_name)
        except ValueError:
            try:
                pdm = ExternalPeakDetectionMethod(peak_detection_method_name)
            except ValueError:
                pdm = GCMSPeakDetectionMethod(peak_detection_method_name)

        # if neither Peak, ExternalPeakDM, or GCMSPDM let ValueError propagate
        pickle_dict = {pdm: self.model_by_pdm[pdm]}
        if use_buffer:
            buffer = BytesIO()
            joblib.dump(pickle_dict, buffer)
            result = buffer.getvalue()
            buffer.close()
        else:
            joblib.dump(pickle_dict, path_to_save)
        return result


    def export_prediction_models(self, path_to_save, use_buffer=False):
        result = None
        if use_buffer:
            buffer = BytesIO()
            joblib.dump(self.model_by_pdm, buffer)
            result = buffer.getvalue()
            buffer.close()
        else:
            joblib.dump(self.model_by_pdm, path_to_save)
        return result



    # @staticmethod
    # def multiclass_prediction_evaluation(model, classes_valid, probabilites, binary_labels):
    #     """
    #     evaluate the prediction of the classes of the test set for a multi class case with false and
    #     true positive rate, and micro and macro averaged ROC-AUC, respectively
    #     :param model: trained model
    #     :param probabilities: prediction probabilities
    #     :param classes_valid: class labels of the validation set
    #     :param binary_labels: binary class labels
    #     :return: AUC (micro, macro, class specific); SEN (micro, macro, class specific), FPR of splits, TPR of splits
    #     """
    #     false_positive_rates_of_splits, true_positive_rates_of_splits = [], []
    #     sensitivities_micro, sensitivities_macro = [], []
    #     false_positive_rate_classes = dict()
    #     sensitivities_classes = dict()
    #     area_under_curve_classes = dict()
    #     for i in range(binary_labels.shape[1]):
    #         false_positive_rate_classes[i] = []
    #         sensitivities_classes[i] = []
    #         area_under_curve_classes[i] = []
    #     areas_under_curve_micro, areas_under_curve_macro = [], []
    #     false_positive_rate = np.linspace(0, 1, 100)
    #     if isinstance(model, RandomForestClassifier):
    #         for i in range(len(probabilites)):
    #             false_positive_rate_micro, true_positive_rate_micro, thresholds = roc_curve(classes_valid[:, i],
    #                                                                                         probabilites[i][:, 1])
    #             areas_under_curve_micro.append(roc_auc_score(classes_valid[:, i], probabilites[i][:, 1], average='micro'))
    #             areas_under_curve_macro.append(roc_auc_score(classes_valid[:, i], probabilites[i][:, 1], average='macro'))
    #             # get the micro sensitivities
    #             sensitivities_micro.append(interp(false_positive_rate, false_positive_rate_micro,
    #                                               true_positive_rate_micro))
    #             sensitivities_micro[-1][0] = 0.0
    #             false_positive_rates_of_splits.append(false_positive_rate_micro)
    #             true_positive_rates_of_splits.append(true_positive_rate_micro)
    #         for label in range(binary_labels.shape[1]):
    #             false_positive_rate_class, true_positive_rate_class, thresholds = roc_curve(classes_valid[:, label],
    #                                                                                         probabilites[label][:, 1])
    #             area_under_curve_classes[label].append(roc_auc_score(classes_valid[:, label], probabilites[label][:, 1]))
    #
    #             false_positive_rate_classes[label].extend(false_positive_rate_class)
    #             sensitivities_classes[label].append(interp(false_positive_rate,
    #                                                        false_positive_rate_class, true_positive_rate_class))
    #             sensitivities_classes[label][-1][0] = 0.0
    #             # Get the macro sensitivity
    #             sensitivities_macro.append(interp(false_positive_rate, false_positive_rate_class,true_positive_rate_class))
    #             sensitivities_macro[-1][0] = 0.0
    #
    #     elif isinstance(model, OneVsRestClassifier):
    #         false_positive_rate_micro, true_positive_rate_micro, thresholds = roc_curve(classes_valid.ravel(),
    #                                                                                     probabilites.ravel())
    #         areas_under_curve_micro.append(roc_auc_score(classes_valid, probabilites, average='micro'))
    #         areas_under_curve_macro.append(roc_auc_score(classes_valid, probabilites, average='macro'))
    #
    #         # get the micro sensitivities
    #         sensitivities_micro.append(interp(false_positive_rate, false_positive_rate_micro,
    #                                           true_positive_rate_micro))
    #         sensitivities_micro[-1][0] = 0.0
    #         false_positive_rates_of_splits.append(false_positive_rate_micro)
    #         true_positive_rates_of_splits.append(true_positive_rate_micro)
    #         for label in range(binary_labels.shape[1]):
    #             false_positive_rate_class, true_positive_rate_class, thresholds = roc_curve(classes_valid[:, label],
    #                                                                                           probabilites[:, label])
    #             area_under_curve_classes[label].append(roc_auc_score(classes_valid[:, label], probabilites[:, label]))
    #
    #             false_positive_rate_classes[label].extend(false_positive_rate_class)
    #             sensitivities_classes[label].append(interp(false_positive_rate,
    #                                                        false_positive_rate_class, true_positive_rate_class))
    #             sensitivities_classes[label][-1][0] = 0.0
    #             # Get the macro sensitivity
    #             sensitivities_macro.append(interp(false_positive_rate, false_positive_rate_class, true_positive_rate_class))
    #             sensitivities_macro[-1][0] = 0.0
    #     return areas_under_curve_micro, areas_under_curve_macro, area_under_curve_classes, sensitivities_micro, sensitivities_macro, sensitivities_classes, false_positive_rates_of_splits, true_positive_rates_of_splits

    # @staticmethod
    # def class_evaluation_wrapper(func, **kwargs):
    #     """
    #     Wraps muticlass_evaluation into dict with keys and values
    #     :return:
    #     """
    #     rv = OrderedDict()
    #     auc_micro, auc_macro, auc_classes, sensitivities_micro, sensitivity_macro, sensitivity_classes, FPR_of_splits, TPR_of_splits = func(**kwargs)
    #     rv['auc_micro'] = auc_micro
    #     rv['auc_macro'] = auc_macro
    #     rv['auc_classes'] = auc_classes
    #     # rv['sensitivity_classes'] = sensitivity_classes
    #     # rv['sensitivities_micro'] = sensitivities_micro
    #     rv['fpr_of_splits'] = FPR_of_splits
    #     rv['tpr_of_splits'] = TPR_of_splits
    #     return rv

    @staticmethod
    def train_random_forest_classifier(trainings_matrix, trainings_labels, original_labels, classifier='OneVsRest', n_estimators=2000, min_samples_split=2, max_features='auto'):
        """
        :param trainings_matrix: trainings set
        :param trainings_labels: class labels of the trainings set
        :param classifier: the manner in which the RF is trained, 'AllVsAll' or 'OneVsRest'
        :param n_estimators: number of the estimators the random forest is trained with
        :param min_samples_split: minimum samples per split
        :param max_features: The number of features to consider when looking for the best split
        :return:  trained model
        """
        if classifier == 'OneVsRest':
            model = OneVsRestClassifier(
                RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                       max_features=max_features, class_weight='balanced', n_jobs=-1,
                                       oob_score=True))
            # TODO implement a AllVsAll p_value option
        else:
            raise NotImplementedError(
                "Classifier {} is not implemented. Choose between either of {}".format(
                    classifier, ['OneVsRest', 'AllVsAll']))

        number_of_classes = len(np.unique(original_labels))
        is_binary_case = number_of_classes == 2
        # train model with full dataset

        if is_binary_case:
            model.fit(trainings_matrix, trainings_labels.ravel())
        else:
            model.fit(trainings_matrix, trainings_labels)
        # from sklearn.utils.validation import check_is_fitted
        # check_is_fitted(model, "estimator_")
        # model.predict_proba(trainings_matrix)
        # pdb.set_trace()
        return model


    @staticmethod
    def cross_validate_binary_model(trainings_matrix, trainings_labels, original_labels, n_splits=2, n_estimators=2000, min_samples_split=2,
                                       max_features='auto', class_weight='balanced', n_jobs=-1, oob_score=True):
        # model needs to be a classification model from sklearn
        number_of_classes = len(np.unique(original_labels))
        is_binary_case = number_of_classes == 2

        if not is_binary_case:
            raise ValueError("Only supporting binary classification")

        # evaluation_metrics = ['average_precision', 'f1', 'recall', 'roc_auc', 'accuracy']

        # cross_validate will also fit the model - can't re-use it to generate the tpr splits
        # generate TPR and FPR lists for ROC curves - set random state
        cv = StratifiedKFold(n_splits=n_splits, random_state=17, shuffle=True) # random state doesnt have an effect as shuffling is disabled
        # tprs, fprs, aucs = defaultdict(list), defaultdict(list), defaultdict(list)
        tprs, fprs, aucs = [], [], []
        tpr_of_splits = []
        fpr_of_splits = []
        evaluations = defaultdict(list)

        # for cv we need to take unary int labels to generate the indices, we use those to select from the binary labels
        for split_no, (train, test) in enumerate(cv.split(trainings_matrix, trainings_labels)):

            # use probas to construct roc measures
            estimation_model = OneVsRestClassifier(
                    RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, n_jobs=n_jobs,
                                       oob_score=oob_score))
            estimation_model.fit(trainings_matrix[train], trainings_labels[train])
            probas_ = estimation_model.predict_proba(trainings_matrix[test])
            predictions_ = estimation_model.predict(trainings_matrix[test])

            # probas_[:, 0] it should be 1 here - not 0
            # fpr, tpr,  _ = metrics.roc_curve(trainings_labels[test], probas_[:, 0])
            fpr, tpr,  _ = metrics.roc_curve(trainings_labels[test], probas_[:, 1])
            evaluations['accuracy'].append(metrics.accuracy_score(trainings_labels[test],predictions_))
            evaluations['f1'].append(metrics.f1_score(trainings_labels[test],predictions_))
            evaluations['recall'].append(metrics.recall_score(trainings_labels[test],predictions_))
            evaluations['average_precision'].append(metrics.average_precision_score(trainings_labels[test],predictions_))

            auc_score = metrics.auc(fpr, tpr)
            evaluations['roc_auc'].append(auc_score)
            aucs.append(auc_score)

            tpr_of_splits.append(tpr.tolist())
            fpr_of_splits.append(fpr.tolist())

        auc_measures = dict()
        auc_measures['tpr_of_splits'] = tpr_of_splits
        auc_measures['fpr_of_splits'] = fpr_of_splits
        auc_measures['auc_of_splits'] = aucs
        # auc_measures['tpr_of_splits_by_class'] = tprs
        # auc_measures['fpr_of_splits_by_class'] = fprs
        # auc_measures['auc_of_splits_by_class'] = aucs
        # auc_measures['mean_tpr'] = mean_tpr

        mean_std_evaluations = {k: (np.mean(v), np.std(v)) for k, v in evaluations.items()}
        auc_measures['mean_auc'], auc_measures['std_auc'] = mean_std_evaluations['roc_auc']

        # save auc measures for plot in evaluations
        # need one cross val roc curve for each class - see http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        mean_std_evaluations['auc_measures'] = auc_measures
        return mean_std_evaluations


    @staticmethod
    def cross_validate_multiclass_model(trainings_matrix, trainings_labels, original_labels, n_splits=2, n_estimators=2000,
                                    min_samples_split=2, max_features='auto', class_weight='balanced', n_jobs=-1,
                                        oob_score=True):
        # model needs to be a classification model from sklearn
        number_of_classes = len(np.unique(original_labels))
        # evaluation_metrics = ['average_precision', 'f1', 'recall', 'roc_auc', 'accuracy']

        # cross_validate will also fit the model - can't re-use it to generate the tpr splits
        # generate TPR and FPR lists for ROC curves - set random state
        cv = StratifiedKFold(n_splits=n_splits, random_state=17, shuffle=True) # random state doesnt have an effect as shuffling is disabled
        # tprs, fprs, aucs = defaultdict(list), defaultdict(list), defaultdict(list)
        tprs, fprs, aucs = [], [], []
        tpr_of_splits = []
        fpr_of_splits = []
        evaluations = defaultdict(list)

        tpr_of_splits_by_class = defaultdict(list)
        fpr_of_splits_by_class = defaultdict(list)
        auc_of_splits_by_class = defaultdict(list)

        # for cv we need to take unary int labels to generate the indices, we use those to select from the binary labels
        for split_no, (train, test) in enumerate(cv.split(trainings_matrix, trainings_labels)):
            # use probas to construct roc measures

            estimation_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, n_jobs=n_jobs, oob_score=oob_score))
            estimation_model.fit(trainings_matrix[train], label_binarize(trainings_labels[train], classes=range(number_of_classes)))
            probas_ = estimation_model.predict_proba(trainings_matrix[test])
            predictions_ = estimation_model.predict(trainings_matrix[test])
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            test_labels = trainings_labels[test]
            binarized_test_labels = label_binarize(test_labels, classes=range(number_of_classes))
            for i in range(number_of_classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(binarized_test_labels[:, i], probas_[:, i])
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                # make sure no arrays used for json seriazability of models
                fpr_of_splits_by_class[i].append(list(fpr[i]))
                tpr_of_splits_by_class[i].append(list(tpr[i]))
                auc_of_splits_by_class[i].append(roc_auc[i])

            evaluations['accuracy'].append(metrics.accuracy_score(binarized_test_labels, predictions_))
            evaluations['f1'].append(metrics.f1_score(binarized_test_labels, predictions_, average="macro"))
            evaluations['recall'].append(metrics.recall_score(binarized_test_labels, predictions_, average="macro"))
            evaluations['average_precision'].append(
                metrics.average_precision_score(binarized_test_labels, predictions_, average="macro"))

            all_fpr_for_split = np.unique(np.concatenate([fpr[i] for i in range(number_of_classes)]))

            mean_tpr_for_split = np.zeros_like(all_fpr_for_split)
            for i in range(number_of_classes):
                mean_tpr_for_split += interp(all_fpr_for_split, fpr[i], tpr[i])

            mean_tpr_for_split /= number_of_classes

            tpr_of_splits.append(mean_tpr_for_split)
            fpr_of_splits.append(all_fpr_for_split)

            fpr['macro'] = all_fpr_for_split
            tpr['macro'] = mean_tpr_for_split
            roc_auc['macro'] = metrics.auc(fpr['macro'], tpr['macro'])

            evaluations['roc_auc'].append(roc_auc['macro'])

            tprs.append(tpr)
            fprs.append(fpr)
            aucs.append(roc_auc)

        # sanitize the np.arrays into lists
        auc_measures = dict()
        auc_measures['tpr_of_splits'] = [list(e) for e in tpr_of_splits]
        auc_measures['fpr_of_splits'] = [list(e) for e in fpr_of_splits]
        auc_measures['auc_of_splits'] = aucs

        # adjust structure so that we have a list for each class, where each element is the tpr for that split
        auc_measures['tpr_of_splits_by_class'] = tpr_of_splits_by_class
        auc_measures['fpr_of_splits_by_class'] = fpr_of_splits_by_class
        auc_measures['auc_of_splits_by_class'] = auc_of_splits_by_class
        # auc_measures['mean_tpr'] = mean_tpr

        mean_std_evaluations = {k: (np.mean(v), np.std(v)) for k, v in evaluations.items()}
        auc_measures['mean_auc'], auc_measures['std_auc'] = mean_std_evaluations['roc_auc']

        # save auc measures for plot in evaluations
        # need one cross val roc curve for each class - see http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        mean_std_evaluations['auc_measures'] = auc_measures
        return mean_std_evaluations


    @staticmethod
    def cross_validate_model(model, trainings_matrix, trainings_labels, original_labels, n_splits=5, n_estimators=35, min_samples_split=2,
                                       max_features='auto', class_weight='balanced', n_jobs=-1, oob_score=True):

        # model needs to be a classification model from sklearn
        number_of_classes = len(np.unique(original_labels))
        is_binary_case = number_of_classes == 2

        evaluation_metrics = ['average_precision']
        metric_differ_multiclass = ['f1', 'recall']
        if is_binary_case:
            evaluation_metrics.extend([m for m in metric_differ_multiclass])
        else:
            evaluation_metrics.extend(["{0}_samples".format(m) for m in metric_differ_multiclass])

        # only apply binarization if neccessary
        if not is_binary_case:
            binarizer = MultiLabelBinarizer(classes=np.sort(np.unique(original_labels)))
            binarizer.fit(np.sort(np.unique(original_labels)))

            # also create int mapping from string labels to
            label_to_int_map = {label: i for i, label in enumerate(np.unique(original_labels))}
            unary_labels = [ele[0] for ele in binarizer.inverse_transform(trainings_labels)]
            unary_int_labels = np.asarray([label_to_int_map[ele] for ele in unary_labels], dtype=int)
        else:
            # label_to_int_map = {label: i for i, label in enumerate(np.unique(trainings_labels))}
            # unary_int_labels = np.asarray([label_to_int_map[ele] for ele in trainings_labels], dtype=int)
            unary_int_labels = trainings_labels


        # from sklearn.utils.multiclass import type_of_target
        # type_of_target(unary_int_labels)

        # roc is computed separately
        if is_binary_case:
            # only difference is ravel() call
            # trainings matrix is empty
            evaluations = cross_validate(model, trainings_matrix, trainings_labels.ravel(), cv=n_splits, scoring=evaluation_metrics, return_train_score=False, )

        else:
            evaluations = cross_validate(model, trainings_matrix, trainings_labels, cv=n_splits, scoring=evaluation_metrics, return_train_score=False, )
            # evaluations = cross_validate(model, trainings_matrix, unary_int_labels, cv=, scoring=evaluation_metrics, return_train_score=False, )
        evaluations.update(cross_validate(model, trainings_matrix, unary_int_labels, cv=n_splits, scoring=['accuracy'], return_train_score=False,))

        # probas = cross_val_predict(model, trainings_matrix, unary_int_labels, cv=5)

        from collections import defaultdict
        tprs, fprs, aucs = defaultdict(list), defaultdict(list), defaultdict(list)
        # tprs, fprs, aucs = dict(), dict(), dict()
        mean_tpr, mean_fpr, mean_auc = dict(), dict(), dict()
        for i in range(number_of_classes):
            tprs[i] = []
            fprs[i] = []
            aucs[i] = []

        # mean_fpr = np.linspace(0, 1, 100)
        tpr_of_splits, fpr_of_splits = [], []

        # generate TPR and FPR lists for ROC curves - set random state
        cv = StratifiedKFold(n_splits=n_splits, random_state=17)
        # for cv we need to take unary int labels to generate the indices, we use those to select from the binary labels
        for train, test in cv.split(trainings_matrix, unary_int_labels):
            # use probas to construct roc measures
            probas_ = model.fit(trainings_matrix[train], trainings_labels[train]).predict_proba(trainings_matrix[test])
            for i in range(number_of_classes):
                # compute roc curve and auc for each class
                if is_binary_case:
                    fpr, tpr, _ = metrics.roc_curve(trainings_labels[test], probas_[:, i])
                else:
                    fpr, tpr, _ = metrics.roc_curve(trainings_labels[test][:, i], probas_[:, i])
                # tprs[i].append(interp(mean_fpr, fpr, tpr))
                # tprs[i][-1][0] = 0.0
                # What do we do when we get NANs? Replacing by zeros is not a good option...
                # fpr[np.isnan(fpr)] = 0.0
                # tpr[np.isnan(tpr)] = 0.0

                tprs[i].append(tpr.tolist())
                fprs[i].append(fpr.tolist())
                roc_auc = metrics.auc(fpr, tpr)
                aucs[i].append(roc_auc.tolist())
                tpr_of_splits.append(tpr.tolist())
                fpr_of_splits.append(fpr.tolist())

        std_auc = dict()
        for i in range(number_of_classes):
            # mean_tpr[i] = np.mean(tprs[i])
            # mean_fpr[i] = np.mean(fprs[i])
            # contains_nans =
            # mean_auc[i] = auc(mean_fpr[i], mean_tpr[i])
            mean_auc[i] = np.mean(aucs[i])
            std_auc[i] = np.std(aucs[i])

        # std_tpr = np.std(tprs, axis=0)

        auc_measures = dict()
        auc_measures['tpr_of_splits'] = tpr_of_splits
        auc_measures['fpr_of_splits'] = fpr_of_splits
        auc_measures['auc_of_splits'] = aucs
        auc_measures['tpr_of_splits_by_class'] = tprs
        auc_measures['fpr_of_splits_by_class'] = fprs
        auc_measures['auc_of_splits_by_class'] = aucs
        # auc_measures['mean_tpr'] = mean_tpr
        auc_measures['mean_auc'] = mean_auc
        auc_measures['std_auc'] = std_auc
        # auc_measures['std_tpr'] = std_tpr

        mean_std_evaluations = {k: (np.mean(v), np.std(v)) for k, v in evaluations.items()}

        # save auc measures for plot in evaluations
        # need one cross val roc curve for each class - see http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        mean_std_evaluations['auc_measures'] = auc_measures
        return mean_std_evaluations


    @staticmethod
    def cross_validate_random_forest_model(trainings_matrix, trainings_labels, original_labels, n_splits=2, classifier_type="OneVsRest", n_estimators=2000, min_samples_split=2, max_features='auto', n_jobs=-1):
        if classifier_type == 'OneVsRest':
            model = OneVsRestClassifier(
                RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                       max_features=max_features, class_weight='balanced', n_jobs=n_jobs,
                                       oob_score=True))
        else:
            raise NotImplementedError(
                "Classifier {} is not implemented. Use 'OneVsRest' instead".format(classifier_type))
        return AnalysisResult.cross_validate_model(model=model, trainings_matrix=trainings_matrix,
                                                   trainings_labels=trainings_labels, original_labels=original_labels,
                                                   n_splits=n_splits, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                       max_features=max_features, class_weight='balanced', n_jobs=n_jobs,
                                       oob_score=True)


    @staticmethod
    def extract_feature_importances(model):
        feature_importances = []
        if isinstance(model, RandomForestClassifier):
            feature_importances.append(model.feature_importances_)
        # importances = self.model.estimator.feature_importances_
        elif isinstance(model, OneVsRestClassifier):
            for estimator in model.estimators_:
                feature_importances.append(estimator.feature_importances_)
        return feature_importances


    # @staticmethod
    # def get_n_smallest_values(dictionary, n):
    #     # based on https://codereview.stackexchange.com/questions/165056/python-3-get-n-greatest-elements-in-dictionary
    #     longest_entries = sorted(dictionary.items(), key=lambda t: len(t[1]))[:n]
    #     return [(key, len(value)) for key, value in longest_entries]

    def extract_best_features_gini(self, model, class_labels, trainings_matrix, peak_coordinates, n_of_reported_peaks, dataset_name, peak_detection_name, add_pvalues=True):
        """
        Extracts the features ranked by gini, Q-value and returns them as DataFrame
        if self.peak_coordinates is set, coordinates are added to resulting DataFrame
        :param model: trained model
        :param class_labels: class labels
        :param trainings_matrix: trainings matrix
        :param peak_coordinates: peak coordinates for peak detection method
        :param n_of_reported_peaks: number of features which will be extracted
        :param dataset_name:
        :param peak_detection_name:
        :param add_pvalues: compute / re-use FDR corrected p_values and save in DataFrame in column 'corrected_p_values'
        :return:
        """
        statistics_best_features = {}
        feature_importances = AnalysisResult.extract_feature_importances(model)

        is_binary = (len(np.unique(class_labels)) == 2)

        # Sort the features in order of decreasing importances
        if is_binary or isinstance(model, RandomForestClassifier):
            # best features are peak IDS - features are ordered as the the trainings matrix
            # sort in descending order
            corrected_pvals_df = self.get_pvalues_df()
            pvals = corrected_pvals_df.loc[corrected_pvals_df['peak_detection_method_name'] == peak_detection_name, corrected_pvals_df.columns]
            pvals['1-corrected_p_values'] = 1 - pvals.loc[:, 'corrected_p_values']
            gini_df = pd.DataFrame({'gini_decrease': feature_importances[0],
                                    'peak_id': trainings_matrix.columns.values,})
            merged = pd.merge(pvals, gini_df, on='peak_id')
            sliced_sorted_merged = merged.sort_values(by=['gini_decrease', '1-corrected_p_values'], ascending=False,)[:n_of_reported_peaks]

            # don't have peak corrdinates when importing feature matrix - so need to skip those keys here and when preparing for template
            if self.peak_coordinates[peak_detection_name] is None:
                best_df = sliced_sorted_merged[['peak_id', 'peak_detection_method_name', 'gini_decrease', 'corrected_p_values']]
            else:
                best_df = sliced_sorted_merged[['peak_id', 'peak_detection_method_name', 'gini_decrease', 'corrected_p_values', 'inverse_reduced_mobility', 'radius_inverse_reduced_mobility', 'retention_time', 'radius_retention_time']]
            statistics_best_features['overall'] = best_df

        else:  # more than two class labels or OneVsRestClassifier -->
            # dont recompute p_values for each pdmn and class combination iteration --> cache them
            # p_values are not available in multiclass classification - would need to recompute for 1 vs rest
            # cannot add p_value for more than
            corrected_pvals_df = self.get_pvalues_df()
            # adresses issue #3
            for class_label, feature_importance in zip(np.unique(class_labels), feature_importances):
                comparison_str = f"{class_label}_VS_Rest"
                pvals_current_class = corrected_pvals_df[
                        np.logical_and(corrected_pvals_df['peak_detection_method_name'] == peak_detection_name,
                                       corrected_pvals_df['class_comparison'] == comparison_str)]
                pvals_current_class['1-corrected_p_values'] = 1 - pvals_current_class['corrected_p_values']
                gini_df = pd.DataFrame({'gini_decrease': feature_importances[0], 'peak_id': trainings_matrix.columns.values, })

                merged = pd.merge(pvals_current_class, gini_df, on='peak_id')
                sliced_sorted_merged = merged.sort_values(by=['gini_decrease', '1-corrected_p_values'],
                                                          ascending=False, )[:n_of_reported_peaks]

                if self.peak_coordinates[peak_detection_name] is None:
                    best_df = sliced_sorted_merged[
                        ['peak_id', 'peak_detection_method_name', 'gini_decrease', 'corrected_p_values']]
                else:
                    best_df = sliced_sorted_merged[
                    ['peak_id', 'peak_detection_method_name', 'gini_decrease', 'corrected_p_values',
                     'inverse_reduced_mobility', 'radius_inverse_reduced_mobility', 'retention_time',
                     'radius_retention_time']]

                statistics_best_features[comparison_str] = best_df
            # for class_label, feature_importance in zip(class_labels, feature_importances):
            #     # sliced_by_importance = sorted_by_importance[:n_of_reported_peaks]
            #     # best_feature_index = trainings_matrix.columns[sliced_by_importance]
            #     gini_df = pd.DataFrame({'gini_decrease': feature_importances[0], 'peak_id': trainings_matrix.columns.values})
            #     sliced_sorted_merged = gini_df.sort_values(by='gini_decrease', ascending=False, )[:n_of_reported_peaks]
            #     statistics_best_features[class_label] = sliced_sorted_merged

        # --> combine the dataframes into a single dataframe after return

        header_str = "#This file contains statistical results from evaluating the dataset {}:\n".format(dataset_name)
        intro_str = \
            "\nThe {}-method was applied as peak detection. The statistics of the {} best features for".format(
                peak_detection_name, n_of_reported_peaks)
        # we train one model for each class - except for when it's a binary problem
        # therefore we need to append to the statistics file if we want to keep all performances
        for i, key in enumerate(statistics_best_features.keys()):
            class_model_statistics_str = "{} class {} are:\n{}".format(intro_str, key, statistics_best_features[key])
        return statistics_best_features


    @staticmethod
    def _apply_function_per_peak_detection_method_dict_OvR(intensities_by_peak_detection_method_by_class, func):
        """
        Apply a function on the intensity values for all peak_detection_methods, pairwise classes and all PeakIDs
        Pairwise classes doesnt mean all binary combinations - but instead all OneVsRest combinations - so one p_value for each class for each peak
        :param intensities_by_peak_detection_method_by_class: Should be formed like d['peak_detection'][class_label] = {peak_id: list(intensities)}
        :param func: Should take two lists or np.arrays
        :return:
        """
        #
        func_by_peak_detection_by_peak_id_by_class = dict()
        for peak_detection, class_dict in intensities_by_peak_detection_method_by_class.items():
            class_labels = [cl for cl in class_dict.keys()]
            peak_ids = [pi for pi in class_dict[class_labels[0]].keys()]
            func_by_peak_by_class = {peak_id: {} for peak_id in peak_ids}
            # create combinations - leaving out one label

            is_binary_case = len(class_labels) == 2
            if is_binary_case:
                l1 = np.unique(class_labels)[0]
                l2 = np.unique(class_labels)[1]
                for peak_id in peak_ids:
                    peaks_l1 = class_dict[l1][peak_id]
                    peaks_l2 = class_dict[l2][peak_id]
                    comparison_str = "{}_VS_{}".format(l1, l2)
                    try:
                        func_by_peak_by_class[peak_id][comparison_str] = func(peaks_l1, peaks_l2)
                    except ValueError:
                        print(peak_id)
                        # something went wrong - so do
                        # problem when we have a peak that has a uniform value - undefined p_value - so assign 1.0 as p_value
                        func_by_peak_by_class[peak_id][comparison_str] = (1.0, 1.0)
                print(f"Computing {peak_detection} pvalues for comparison_str {comparison_str}")
            else:
                for l2 in combinations(class_labels, len(class_labels)-1):
                    # left out label is our _l1_ VsRest label
                    l1 = set(l2).symmetric_difference(set(class_labels)).pop()
                    for peak_id in peak_ids:
                        peaks_l1 = class_dict[l1][peak_id]
                        peaks_l2 = []
                        for rest_label in l2:
                            peaks_l2.extend(class_dict[rest_label][peak_id])
                        comparison_str = "{}_VS_{}".format(l1, "Rest")
                        # apply the function to all possible pairwise combinations of the classes --> threshold and p-value
                        func_by_peak_by_class[peak_id][comparison_str] = func(peaks_l1, peaks_l2)
                    print(f"Computing {peak_detection} pvalues for comparison_str {comparison_str}")

            func_by_peak_detection_by_peak_id_by_class[peak_detection] = func_by_peak_by_class
        return func_by_peak_detection_by_peak_id_by_class


    @staticmethod
    def _apply_function_per_peak_detection_method_dict_AvA(intensities_by_peak_detection_method_by_class, func):
        """
        Apply a function on the intensity values for all peak_detection_methods, pairwise classes and all PeakIDs
        Pairwise means all binary combinations - AllvsALL combiation
        :param intensities_by_peak_detection_method_by_class: Should be formed like d['peak_detection'][class_label] = {peak_id: list(intensities)}
        :param func: Should take two lists or np.arrays
        :return:
        """
        #
        func_by_peak_detection_by_peak_id_by_class = dict()
        for peak_detection, class_dict in intensities_by_peak_detection_method_by_class.items():
            class_labels = [cl for cl in class_dict.keys()]
            peak_ids = [pi for pi in class_dict[class_labels[0]].keys()]
            func_by_peak_by_class = {peak_id: {} for peak_id in peak_ids}
            # create combinations
            for l1, l2 in combinations(class_labels, 2):
                for peak_id in peak_ids:
                    peaks_l1 = class_dict[l1][peak_id]
                    peaks_l2 = class_dict[l2][peak_id]
                    comparison_str = "{}_VS_{}".format(l1, l2)
                    # apply the function to all possible pairwise combinations of the classes --> threshold and p-value
                    func_by_peak_by_class[peak_id][comparison_str] = func(peaks_l1, peaks_l2)

            func_by_peak_detection_by_peak_id_by_class[peak_detection] = func_by_peak_by_class
        return func_by_peak_detection_by_peak_id_by_class


    @staticmethod
    def permutation_test(dictionary, n_permutations):
        # permutation test with 100 permutations, based on monte calro method
        def exact_mc_perm_test(xs, ys, nmc=n_permutations):
            n, k = len(xs), 0
            diff = np.abs(np.mean(xs) - np.mean(ys))
            zs = np.concatenate([xs, ys])
            for j in range(nmc):
                np.random.shuffle(zs)
                k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
            return k / nmc
        return AnalysisResult._apply_function_per_peak_detection_method_dict_OvR(
            dictionary, exact_mc_perm_test)


    @staticmethod
    def mann_whitney_u_per_dict(intensities_by_peak_detection_method_by_class):
        return AnalysisResult._apply_function_per_peak_detection_method_dict_OvR(
            intensities_by_peak_detection_method_by_class, mannwhitneyu)


    @staticmethod
    def _correct_multiple_testing_for_mann_whitney_u(mwu_by_peak_detection_by_peak_id_by_class, method="fdr_bh",
                                                    alpha=0.05):
        """
        :param mwu_by_peak_detection_by_peak_id_by_class:
        :param method: string
            Method used for testing and adjustment of pvalues. Can be either the
            full name or initial letters. Available methods are ::

            `bonferroni` : one-step correction
            `sidak` : one-step correction
            `holm-sidak` : step down method using Sidak adjustments
            `holm` : step-down method using Bonferroni adjustments
            `simes-hochberg` : step-up method  (independent)
            `hommel` : closed method based on Simes tests (non-negative)
            `fdr_bh` : Benjamini/Hochberg  (non-negative)
            `fdr_by` : Benjamini/Yekutieli (negative)
            `fdr_tsbh` : two stage fdr correction (non-negative)
            `fdr_tsbky` : two stage fdr correction (non-negative)
        :param alpha:
        :return: pd.DataFrame
        """
        comp_dfs = []
        for peak_detection, mwu_by_peak_by_class in mwu_by_peak_detection_by_peak_id_by_class.items():
            peak_ids = [pi for pi in mwu_by_peak_by_class.keys()]
            comparison_strs = [cs for cs in mwu_by_peak_by_class[peak_ids[0]]]

            for comp_str in comparison_strs:
                all_p_vals_by_comparison = []
                for p_id in peak_ids:
                    all_p_vals_by_comparison.append(mwu_by_peak_by_class[p_id][comp_str][1])

                # rejected, corrected_p = fdrcorrection0(all_p_vals_by_comparison, alpha=benjamini_hochberg_alpha)
                # instead multipletests() could be used, provides other correction methods, such as bonferoni and more
                tuple = multipletests(pvals=all_p_vals_by_comparison, method=method, alpha=alpha)
                try:
                    rejected, corrected_p = tuple[0], tuple[1]
                except ValueError:
                    raise NotImplementedError("{} not supported in correction for multiple testing. Please use 'fdr_bh' instead.".format(method))

                comp_df = pd.DataFrame(data=(rejected), columns=['null_hyp_rejected'])
                comp_df['corrected_p_values'] = corrected_p
                comp_df['peak_id'] = peak_ids
                comp_df['raw_p_values'] = all_p_vals_by_comparison
                comp_df['class_comparison'] = [comp_str] * len(all_p_vals_by_comparison)
                comp_df['peak_detection_method_name'] = [peak_detection] * len(all_p_vals_by_comparison)
                comp_dfs.append(comp_df)
        df_for_p_vals = pd.concat(comp_dfs)
        return df_for_p_vals


    @staticmethod
    def benjamini_hochberg_for_mann_whitney_u(mwu_by_peak_detection_by_peak_id_by_class, benjamini_hochberg_alpha=0.05):
        return AnalysisResult._correct_multiple_testing_for_mann_whitney_u(mwu_by_peak_detection_by_peak_id_by_class,
                                                                          method="fdr_bh",
                                                                          alpha=benjamini_hochberg_alpha)


    @staticmethod
    def intensities_for_features(peak_coordinates, measurements, feature_set_by_pdm_name, trainings_matrix_dict=dict(), class_label_dict=dict()):
        """
        Extract peak intensities form measurements / trainings_matrix
        If featurelist is defined only extract coordinates for it's elements as column names
        Expects class labels to be set for measurements
        extracts maximum intensity from peak window
        :param peak_coordinates:
        :param measurements:
        :param feature_list:
        :return:
        """
        # now filters by feature_names
        intensities_by_peak_detection_method_by_class = dict()
        for peak_detection_method_name, coordinates in peak_coordinates.items():
            # get columns names from trainings_matrix - so we only look at peak_coords that match the feature reduction
            potential_peak_id_set = feature_set_by_pdm_name[peak_detection_method_name]
            if not potential_peak_id_set:
                # if we get an empty feature set default to all peak_id_values
                potential_peak_id_set = set(coordinates.peak_id.values)

            # create empty datastructure for analysis with Man-Whitney-U test
            if measurements:
                intensities_of_classes_per_peak_id = {
                    class_label: {p_id: [] for p_id in coordinates.peak_id.values if p_id in potential_peak_id_set} for
                    class_label in np.unique([m.class_label for m in measurements])}
            else:
                intensities_of_classes_per_peak_id = {
                    class_label: {p_id: [] for p_id in coordinates.peak_id.values if p_id in potential_peak_id_set} for
                    class_label in class_label_dict.values()}

            # for each peak_id it holds the list of maximum intensities

            # if we have PeakAlignmentResult - the peakIntensities should already be in the peakAlignmentResult
            # If we have measurements, this is more accurate - but if not we need to take the ones from the PeakDetectionResults
            # PEAX is an edge case - handles it's own normalization and preprocessing - so need to extract from alignmentResult
            if measurements and peak_detection_method_name != "PEAX":
            # if measurements:
                # filter applicable coordinates
                potential_mask = [pid in potential_peak_id_set for pid in coordinates.peak_id.values]
                reduced_coordinates = coordinates[potential_mask]


                # iterate over all peak_coordinates
                for _, row in reduced_coordinates.iterrows():


                    # get all selection ranges
                    peak_id = row['peak_id']
                    center_irm = row['inverse_reduced_mobility']
                    radius_irm = row['radius_inverse_reduced_mobility']
                    radius_rt = row['radius_retention_time']
                    center_rt = row['retention_time']
                    irm_selection_window = center_irm - radius_irm, center_irm + radius_irm
                    rt_selection_window = center_rt - radius_rt, center_rt + radius_rt
                    for m in measurements:
                        # select by window
                        window_by_irm = m.df.iloc[(m.df.index >= irm_selection_window[0]) & (m.df.index <= irm_selection_window[1])].T
                        # further select from window matchig retention time - then get the max intensity
                        max_intensity = np.max(window_by_irm.iloc[
                                               (np.asarray(window_by_irm.index, dtype=float) >= rt_selection_window[0])
                                           & (np.asarray(window_by_irm.index, dtype=float) <= rt_selection_window[1])
                                               ].max())
                        intensities_of_classes_per_peak_id[m.class_label][peak_id].append(max_intensity)


            # extract intensities from peakAlignmentResult instead of from preprocessed measurements
            else:
                current_feature_matrix = trainings_matrix_dict[peak_detection_method_name]
                # class label_dict is ordered, so items will traverse in correct order
                for measurement_name, class_label in class_label_dict.items():
                    for peak_id in potential_peak_id_set:
                        max_intensity = \
                        current_feature_matrix.loc[current_feature_matrix.index == measurement_name, peak_id].values[0]

                        intensities_of_classes_per_peak_id[class_label][peak_id].append(max_intensity)

            intensities_by_peak_detection_method_by_class[peak_detection_method_name] = intensities_of_classes_per_peak_id
        return intensities_by_peak_detection_method_by_class


    @staticmethod
    def intensities_for_features_custom(feature_set_by_pdm_name, trainings_matrix_dict=dict(), class_label_dict=dict()):
        """
        If featurelist is defined only extract coordinates from them
        Expects class labels to be set for measurements
        extracts maximum intensity from peak window
        :param feature_set_by_pdm_name:
        :return:
        """
        # now filters by feature_names
        intensities_by_peak_detection_method_by_class = dict()
        for peak_detection_method_name, df in trainings_matrix_dict.items():
            # get columns names from trainings_matrix - so we only look at peak_coords that match the feature reduction
            potential_peak_id_set = feature_set_by_pdm_name[peak_detection_method_name]

            # create empty datastructure for analysis with Man-Whitney-U test
            intensities_of_classes_per_peak_id = {
                    class_label: {p_id: [] for p_id in df.columns if p_id in potential_peak_id_set} for
                    class_label in class_label_dict.values()}

            # extract intensities from feature matrix

            current_feature_matrix = df
            # class label_dict is ordered, so items will traverse in correct order
            for measurement_name, class_label in class_label_dict.items():
                for peak_id in potential_peak_id_set:
                    max_intensity = current_feature_matrix.loc[current_feature_matrix.index == measurement_name, peak_id].values[0]
                    intensities_of_classes_per_peak_id[class_label][peak_id].append(max_intensity)

            intensities_by_peak_detection_method_by_class[peak_detection_method_name] = intensities_of_classes_per_peak_id
        return intensities_by_peak_detection_method_by_class


    # @staticmethod
    # def intensities_per_class_for_best_features_mann_whitney(best_features, measurements):
    #     """
    #     Extracting the intensities of the best feature for each measurement and group them by the classes. Additionally
    #     the Mann-Whitney-U test is applied to compare the intensities of the classes
    #
    #     intensities_per_class_and_best_feature will contain the maximum intensities of all measurements, separated by class label
    #     :param best_features: Best features detemined by the classifier
    #     :param measurements: Measurements
    #     :return:
    #     """
    #     x_steps, y_steps = MccImsAnalysis.compute_window_size()
    #     intensities_best_features = dict()
    #     mann_whitney_u = dict()
    #     for key in best_features.keys():
    #         intensities_best_features[key] = dict()
    #         mann_whitney_u[key] = dict()
    #         # for feature_number in range(best_features[key].shape[0]):
    #         for feature_number in best_features[key].index:
    #             intensities_of_classes = {class_label: [] for class_label in
    #                                       np.unique([m.class_label for m in measurements])}
    #             # Extract the window of the feature (inverse_reduced_mobility and retention_time)
    #             # the window size is defined by our probe clustering approach, using 2 steps as width and height
    #             index_inverse_reduced_mobility = np.argmax(x_steps[x_steps <= best_features[key].loc[feature_number, 'inverse_reduced_mobility']])
    #             inverse_reduced_mobility_window = x_steps[index_inverse_reduced_mobility: index_inverse_reduced_mobility + 2]
    #             index_retention_time = np.argmax(y_steps[y_steps <= best_features[key].loc[feature_number, 'retention_time']])
    #             retention_time_window = y_steps[index_retention_time: index_retention_time + 2]
    #             # check if the inverse reduced mobility is not the upper limit
    #             for measurement in measurements:
    #                 if len(inverse_reduced_mobility_window) == 2:
    #                     intensities_in_window = measurement.df[(measurement.df.index >= inverse_reduced_mobility_window[0]) & (measurement.df.index <= inverse_reduced_mobility_window[1])].T
    #                     # string comparison problem, we have strings in our imported dataframe index! - we have to cast it to float
    #                     if len(retention_time_window) == 2: #  check if the retention time is not the upper limt
    #                         intensities_in_window = intensities_in_window[
    #                             (np.asarray(intensities_in_window.index, dtype=float) >= retention_time_window[0]) &
    #                             (np.asarray(intensities_in_window.index, dtype=float) <= retention_time_window[1])].T
    #                     #  if the retention time is the upper limit
    #                     else:
    #                         intensities_in_window = intensities_in_window[
    #                             (np.asarray(intensities_in_window.index, dtype=float) >= retention_time_window[0])].T
    #                     max_intensity = np.max(intensities_in_window.max())
    #                     intensities_of_classes[measurement.class_label].append(max_intensity)
    #                 else: # if the inverse reduced mobility is the upper limit --> no second value to define the window is available
    #                     intensities_in_window = measurement.df[(measurement.df.index >= inverse_reduced_mobility_window[0])].T
    #                     if len(retention_time_window) == 2: # if the retention time is not the upper limit
    #                         intensities_in_window = intensities_in_window[
    #                             (np.asarray(intensities_in_window.index, dtype=float) >= retention_time_window[0]) &
    #                             (np.asarray(intensities_in_window.index, dtype=float) <= retention_time_window[1])].T
    #                     else: # if the retention time is the upper limit
    #                         intensities_in_window = intensities_in_window[
    #                             (np.asarray(intensities_in_window.index, dtype=float) >= retention_time_window[0])].T
    #                     max_intensity = np.max(intensities_in_window.max())
    #                     intensities_of_classes[measurement.class_label].append(max_intensity)
    #
    #             # get the keys of the dictionary intensities of classes as a list
    #             list_of_classes = [*intensities_of_classes]
    #             mann_whitney_u_of_best_feature = dict()
    #             # what are these while loops for?
    #             # n = 0
    #             for n in range(len(list_of_classes)):#while n < len(list_of_classes) - 1:
    #                 x = 1
    #                 while n+x <= len(list_of_classes) - 1:
    #                     comparison = "{}_{}".format(list_of_classes[n], list_of_classes[n+x])
    #                     # apply the mann whitney u test to all possible pairwise combinations of the classes --> threshold and p-value
    #                     mann_whitney_u_of_best_feature[comparison] = mannwhitneyu(
    #                         intensities_of_classes[list_of_classes[n]], intensities_of_classes[list_of_classes[n+x]])
    #                     x += 1
    #                 # n += 1
    #             mann_whitney_u[key][best_features[key].loc[feature_number, 'peak_id']] = mann_whitney_u_of_best_feature
    #             intensities_best_features[key][best_features[key].loc[feature_number, 'peak_id']] = intensities_of_classes
    #     return intensities_best_features, mann_whitney_u


    def create_decision_trees(self, **kwargs):
        """
        Create decision trees for all best_features models of evaluation_method, peakdetection_method and save result
        as string representation
        :return:
        """
        defaults = MccImsAnalysis.DEFAULT_PERFORMANCE_MEASURE_PARAMETERS[PerformanceMeasure.DECISION_TREE_TRAINING]
        max_depth = kwargs.get('max_depth', defaults['max_depth'])
        min_samples_leaf = kwargs.get('min_samples_leaf', defaults['min_samples_leaf'])
        min_samples_split = kwargs.get('min_samples_split', defaults['min_samples_split'])

        required_keys = ['performance_measure_name', 'peak_detection_method_name', 'class_comparison']
        assert all([k in set(self.best_features_df.columns.values) for k in required_keys])
        parms = {"max_depth":max_depth, "min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split}
        print(f"Training decision trees with {parms}")
        for performance_measure_name, df_by_eval_method in self.best_features_df.groupby('performance_measure_name'):
            self.decision_tree_per_evaluation_method_and_peak_detection[performance_measure_name] = dict()
            for peak_detection_name, df_by_peak_detection in df_by_eval_method.groupby('peak_detection_method_name'):
                dot_data_by_peak_detection = []

                # for class_comparison_str, df_by_class_comparison in df_by_peak_detection.groupby('class_comparison'):
                #     # save dot_data for visualization as string in dt_model
                #     dt_model, dot_data = self.train_single_decision_tree(df_by_class_comparison, performance_measure_name=performance_measure_name, peak_detection_name=peak_detection_name, class_comparison_str=class_comparison_str)
                #     dot_data_by_peak_detection.append((class_comparison_str, dot_data))
                # save dot_data for visualization as string in dt_model

                # only do single decision tree for multi class problem - not one dt for each vs rest comparison
                for class_comparison_str, df_by_class_comparison in df_by_peak_detection.groupby('class_comparison'):
                    dt_model, dot_data = self.train_single_decision_tree(
                        df_by_class_comparison, performance_measure_name=performance_measure_name,
                        peak_detection_name=peak_detection_name, class_comparison_str=class_comparison_str,
                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_depth=max_depth,
                    )
                    dot_data_by_peak_detection.append((class_comparison_str, dot_data))
                    break

                self.decision_tree_per_evaluation_method_and_peak_detection[performance_measure_name][
                    peak_detection_name] = dot_data_by_peak_detection


    def train_single_decision_tree(self, best_features_selection_df, performance_measure_name, peak_detection_name, class_comparison_str, min_samples_leaf=1, min_samples_split=2, max_depth=5):
        """
        Build a decision_tree with the passed best_features
        :return:
        """
        # remove export functionality here -> save result as string
        # visualization with graphviz and customized with small bars representing the splits
        # Create the trainings matrix of the n best features
        class_labels = np.unique(self.class_labels)
        # get trainings_labels
        feature_names = best_features_selection_df['peak_id'].values

        trainings_matrix = self.trainings_matrix[peak_detection_name].loc[:, feature_names]
        # minimum split to 1/5 of samples, we dont want miniscule splits
        samples_per_class = Counter(self.class_labels)

        # leave unpruned instead of 1/5 of samples
        # minimum_observations = min(samples_per_class.values())
        # decision_tree_model = tree.DecisionTreeClassifier(min_weight_fraction_leaf=0.1)
        # min_samples_split = max(2, int(minimum_observations / 5))
        # use parameters for decision tree training configuration
        decision_tree_model = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_depth=max_depth)

        # replace NaNs with 0
        trainings_matrix.fillna(0, inplace=True)

        decision_tree_model.fit(trainings_matrix, self.class_labels)
        # extended tree.export_graphviz to make a sensible representation for us - using a bar to represent class distribution
        # features: Non rounded, no gini, percentage bar for %samples
        from ..tools.tools import export_graphviz_personalized
        tree.export_graphviz_personalized = export_graphviz_personalized

        dot_data = tree.export_graphviz_personalized(
            decision_tree_model,
            feature_names=feature_names.tolist(),
            class_names=class_labels, out_file=None, proportion=True, filled=True, leaves_parallel=True)

        return decision_tree_model, dot_data


class PredictionModel(object):
    """
    Holds preprocessing and evaluation parameters and model for class prediction
    """

    def __init__(self, preprocessing_params, evaluation_params, scipy_predictor_by_pdm, feature_names_by_pdm, dir_level = "", visualnow_layer_file="", peax_binary_path=""):
        dataset_name = "new_dataset"
        self.analysis = MccImsAnalysis(
            measurements=[], preprocessing_steps=preprocessing_params.keys(),
                                preprocessing_parameters=preprocessing_params,
                                outfile_names=[],
                                dir_level=dir_level,
                                dataset_name=dataset_name,
                                visualnow_layer_file=visualnow_layer_file,
                                peax_binary_path=peax_binary_path)
        self.preprocessing_params = preprocessing_params
        self.evaluation_params = evaluation_params
        self.scipy_predictor_by_pdm = scipy_predictor_by_pdm
        self.feature_names_by_pdm = feature_names_by_pdm


    @staticmethod
    def reconstruct_remove_features(matrices_by_pdm, feature_names_by_pdm):
        matching_matrices = dict()
        # keys = feature_names_by_pdm.keys()
        keys = matrices_by_pdm.keys()
        for pdm in keys:
            if isinstance(pdm, str):
                pdm_name = pdm
            else:
                pdm_name = pdm.name
            matrix = matrices_by_pdm[pdm_name]
            feature_names = feature_names_by_pdm[pdm]
            # all features that are not in the index we need to fill with NA / 0.0
            # select by columns
            different_columns = set(matrix.columns.values.tolist()).difference(set(feature_names))
            columns_to_init_null = different_columns.intersection(set(feature_names))
            # fill columns with 0.0
            if columns_to_init_null:
                assign_dict = {col: 0.0 for col in columns_to_init_null}
                matrix = matrix.assign(**assign_dict)
                print(f"Missing: {columns_to_init_null}, initilizing with 0.0")

            # < pandas 1.x version
            # matching_matrices[pdm] = matrix.loc[:, feature_names]
            # 1.0.1  due to deprecation
            matching_matrices[pdm] = matrix.loc[:, matrix.columns.intersection(feature_names)].reindex(
                columns=feature_names).fillna(0)
            # from IPython import embed;
            # from celery.contrib import rdb;
            # rdb.set_trace()
            # embed()

        return matching_matrices


    def predict(self, measurements):
        # make a new analysis with preprocessing and evaluation parameters
        # return the list of predicted classes for the input
        # create temp dir to write to

        tmp = os.path.join(tempfile.gettempdir(), '.breath/{}'.format(hash(os.times())))
        os.makedirs(tmp)
        outfile_names = [tmp + '/' + m.filename[:-4] + "_out.csv" for m in measurements]
        self.analysis.measurements = measurements
        self.analysis.outfile_names = outfile_names
        self.analysis.preprocess()
        rmtree(tmp, ignore_errors=True)

        self.analysis.align_peaks(file_prefix="")

        #  problem when not using probe clustering!
        #  we need to get the same features as used in training of the scipy_predictor
        df_matching_training = PredictionModel.reconstruct_remove_features(
                self.analysis.peak_alignment_result.dict_of_df, self.feature_names_by_pdm)

        # get labels to assign class and return prediction
        # get evaluation matrix for each chosen peak_detection method
        prediction_by_pdm = dict()
        for pdm in self.analysis.peak_detection_combined:
            # X = self.analysis.peak_alignment_result.dict_of_df[pdm.name]
            # we have NaNs in X_reduced - predict cant handle them
            X_reduced = df_matching_training[pdm.name].fillna(0.0)
            # Import exisitng model and predict classes of unknown data
            # probas = self.scipy_predictor.predict_proba(X)
            # TODO exported predictors should be both a decision tree, and a RandomForestClassifier
            prediction_by_pdm[pdm] = self.scipy_predictor_by_pdm[pdm].predict(X_reduced).tolist()
            # predictor = self.scipy_predictor_by_pdm[pdm]
        return prediction_by_pdm


    def predict_from_peak_detection_results(self, peak_detection_result_dir, class_label_file_name):
        """
        Import peak detection results, align, reduce features and use prediction model for prediction of classes
        :param peak_detection_result_dir: directory in which the peak detection results are found - should have suffix according to peak detection method used
        :return:
        """

        self.analysis.import_results_from_csv_dir(peak_detection_result_dir, class_label_file_name)

        # align_peaks will use the grid and algos passed on initialization
        self.analysis.align_peaks(file_prefix="")

        # make sure features match -
        #  we need to get the same features as used in training of the scipy_predictor
        df_matching_training = PredictionModel.reconstruct_remove_features(
            self.analysis.peak_alignment_result.dict_of_df, self.feature_names_by_pdm)

        # get labels to assign class and return prediction
        # get evaluation matrix for each chosen peak_detection method
        prediction_by_pdm = dict()
        test_matrix_by_pdm = dict()
        for pdm in self.analysis.peak_detection_combined:
            # X = self.analysis.peak_alignment_result.dict_of_df[pdm.name]
            # we have NaNs in X_reduced - predict cant handle them
            X_reduced = df_matching_training[pdm.name].fillna(0.0)
            # Import exisitng model and predict classes of unknown data
            # probas = self.scipy_predictor.predict_proba(X)
            prediction_by_pdm[pdm] = self.scipy_predictor_by_pdm[pdm].predict(X_reduced).tolist()
            test_matrix_by_pdm[pdm] = X_reduced
            # predictor = self.scipy_predictor_by_pdm[pdm]
        return prediction_by_pdm, test_matrix_by_pdm


    def predict_from_feature_matrix(self, pdms, fm_dict_matching_training):
        prediction_by_pdm = dict()
        test_matrix_by_pdm = dict()
        for pdm in pdms:
            # X = self.analysis.peak_alignment_result.dict_of_df[pdm.name]
            # we have NaNs in X_reduced - predict cant handle them
            X_reduced = fm_dict_matching_training[pdm.name].fillna(0.0)
            # Import exisitng model and predict classes of unknown data
            # probas = self.scipy_predictor.predict_proba(X)
            prediction_by_pdm[pdm] = self.scipy_predictor_by_pdm[pdm].predict(X_reduced).tolist()
            test_matrix_by_pdm[pdm] = X_reduced
            # predictor = self.scipy_predictor_by_pdm[pdm]
        return prediction_by_pdm, test_matrix_by_pdm


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)

    return new_func

