import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms

from .ProcessingMethods import GCMSPeakDetectionMethod

"""
Nomenclature: 
    Spectrum: contains all points with a single RT value
    TIS: total ion spectrum, sum of intensity over all spectra
    XIC: extracted ion chromatogram - basically a subset of ms1 spectrum with smaller m/z range
    BPC: Base peak chromatogram - contains mmost intense signal for each RT across all m/z
    Isotope trace: signal that is produced by single ion of a single analute at a particvular charge state
    isotopic envelope trace: group of isotropic traces produced by a single analyte at a particular charge state
    Peak, feature, chromatogram extremely overloaded
"""
class UnsupportedGCMSFiletypeError(ValueError):
    """
    Error when not mzML or mzXML format used
    """


def load_ms_file(ms_experiment, path_to_file):
    """
    Load mzML / mzXML file into given ms_experiment object, filter to MSLevel = 1 - as only those supported for peakFinders
    :param ms_experiment:
    :param path_to_file:
    :param set_options:
    :return:
    """
    options = oms.PeakFileOptions()
    options.setMSLevels([1])  # MS1 is comparable to mcc-ims input
    # RuntimeError: FeatureFinder can only operate on MS level 1 data. Please do not use MS/MS data

    # support mzXML and mzML -file
    suffix = Path(path_to_file).suffix.lower()
    if suffix == ".mzxml":
        fh = oms.MzXMLFile()
    elif suffix == ".mzml":
        fh = oms.MzMLFile()
    else:
        raise UnsupportedGCMSFiletypeError("Unsupported filetype. Only mzXML and mzML format is supported.")

    fh.setOptions(options)
    # load data into experiment

    fh.load(str(path_to_file), ms_experiment)  # if problems loading - let pyopenms error bubble up
    ms_experiment.updateRanges()

    # get keys function loads the return into an empty list - very straightforward to guess
    # k = []
    # ms_experiment[0].getKeys(k)
    # ms_experiment[1].getKeys(k)
    # ms_experiment[2].getKeys(k)
    # ms_experiment[3].getKeys(k)
    # list_functions(ms_experiment[0])
    #
    # ms_experiment[0].getAcquisitionInfo()#
    #
    # oms.CachedmzML.store("myCache.mzML", ms_experiment)
    #
    # # Now load data
    # cfile = oms.CachedmzML()
    # oms.CachedmzML.load("myCache.mzML", cfile)
    #
    # meta_data = cfile.getMetaData()
    # meta_data.metaRegistry()
    # help(meta_data)
    # list_functions(meta_data)
    # meta_data.getKeys(k)
    #
    #
    # k = []
    # list_functions(\
    #     ms_experiment[0].getDataProcessing()[0].getMetaValue(k))
    # print(k)
    # list_functions(fh.getOptions())
    # try centroided approach to pickPeaks
    #   if not centroided,
    #       apply smoothing and peak detection to form centroided data = peakMap / FeatureXML file
    return ms_experiment


def detect_peaks_gcms_centroid(ms_experiment, parameters, debug=False):
    """
    Applicable to centroided experiments, also see https://abibuilder.informatik.uni-tuebingen.de/archive/openms/Documentation/nightly/html/a16103.html
    :param ms_experiment:
    :param parameters:
    :return:
    """
    print(f"Detecting peaks with {GCMSPeakDetectionMethod.CENTROIDED}")
    ff = oms.FeatureFinder()

    if not debug:
        ff.setLogType(oms.LogType.NONE)
    else:
        ff.setLogType(oms.LogType.CMD)

    # Run the feature finder
    name = "centroided"
    pdm_name = GCMSPeakDetectionMethod.CENTROIDED.name
    parameters['detection_mode'] = name
    parameters['pdm_name'] = pdm_name
    # name = parameters['detection_mode']
    features = oms.FeatureMap()
    seeds = oms.FeatureMap()
    ff_params = oms.FeatureFinder().getParameters(name)
    ff.run(name, ms_experiment, features, ff_params, seeds)
    # features.setUniqueIds()
    features.ensureUniqueId()
    fh = oms.FeatureXMLFile()
    feature_storage_path = f"{parameters['input_filename']}_output.featureXML"
    fh.store(feature_storage_path, features)
    parameters['feature_storage'] = feature_storage_path
    print("Found", features.size(), "features")
    return parameters


def detect_peaks_gcms_isotopewavelet(ms_experiment, parameters, debug=False):
    """
    Use isotop wavelet to process raw data - can perform poorly on centroided data
        - also see https://abibuilder.informatik.uni-tuebingen.de/archive/openms/Documentation/nightly/html/a16105.html
    TODO use "corrected" intensity_type
    :param ms_experiment:
    :param parameters:
    :return:
    """
    print(f"Detecting peaks with {GCMSPeakDetectionMethod.ISOTOPEWAVELET}")
    ff = oms.FeatureFinder()
    if not debug:
        ff.setLogType(oms.LogType.NONE)
    else:
        ff.setLogType(oms.LogType.CMD)

    # Run the feature finder
    name = "isotope_wavelet"
    pdm_name = GCMSPeakDetectionMethod.ISOTOPEWAVELET.name
    parameters['detection_mode'] = name
    parameters['pdm_name'] = pdm_name
    # name = parameters['detection_mode']
    features = oms.FeatureMap()
    seeds = oms.FeatureMap()
    ff_params = ff.getParameters(name)

    # complains about "the extremal length of the wavelet is larger (47661) than the number of data points"
    # wavelet_length is defined by mz_cutoff / min_spacing
    # hr_data must be true if high-resolution data (orbitrap, FTICR)
    # hr_data parameter for isotopewavelet function
    is_hr_data = parameters.get("hr_data", False)
    if is_hr_data:
        hr_key = b"hr_data"  # hr_data takes extremely long - >= 2h per measurement of (!)32MB - there are way larger spectra...
        ff_params.setValue(hr_key, b"true")

    ff.run(name, ms_experiment, features, ff_params, seeds)
    features.setUniqueIds()
    fh = oms.FeatureXMLFile()
    feature_storage_path = f"{parameters['input_filename']}_output.featureXML"
    fh.store(feature_storage_path, features)
    parameters['feature_storage'] = feature_storage_path
    print("Found", features.size(), "features")
    return parameters


# two peak pickers - PeakPickerWavelet and PeakPickerHiRes
def detect_peaks_gcms_peak_picker_wavelet(ms_experiment, parameters):
    """
    Use isotop wavelet to process raw data - can perform poorly on centroided data
        https://abibuilder.informatik.uni-tuebingen.de/archive/openms/Documentation/nightly/html/a16159.html
    TODO use "corrected" intensity_type
    :param ms_experiment:
    :param parameters:
    :return:
    """
    #set estimate_peak_width to true


    # Run the feature finder
    name = "peak_picker_wavelet"
    parameters['detection_mode'] = name


    # outdated code in https://github.com/OpenMS/pyopenms-extra/blob/master/src/examples/peakpicker_scipyFFT.py

    # pick spectrum
    ff = oms.FeatureFinder()
    ff.setLogType(oms.LogType.NONE)

    algo_name = oms.FeatureFinderAlgorithmIsotopeWavelet().getProductName()
    # picker = oms.FeatureFinderAlgorithmIsotopeWavelet()

    feature_map = oms.FeatureMap()
    seeds = oms.FeatureMap()

    # seeds = FeatureMap()
    algo_params = ff.getParameters(algo_name)
    ff.run(algo_name, ms_experiment, feature_map, algo_params, seeds)
    feature_map.setUniqueIds()

    parameters = default_store_feature_xml(feature_map, parameters)

    return parameters


def filter_feature_xmls(dir, name_list=[], include_aligned=0):
    """
    Returns list of feature xmls that are not aligned yet , so ending with "ML_output.featureXML"
    :param dir:
    :return:
    """
    if include_aligned:
        suffix_to_check = get_default_feature_xml_storage_suffix()  # includes aligned suffixes
    else:
        suffix_to_check = "ML_output.featureXML"  # only unaligned featureXMLS

    if name_list:
        matching_feature_xmls = [fn for fn in name_list if str(fn).endswith(suffix_to_check)]
    else:
        matching_feature_xmls = glob.glob(str(dir) + f"/*{suffix_to_check}")
    return sorted(matching_feature_xmls)


# needs to match class_labels with the feature_xml_lis - otherwise problem with feature_matrix index
def align_feature_xmls(feature_xml_lis, consensus_map_out_path="", class_label_dict={}):
    """
    first apply pose clustering to include all features maps
      next link/group them across all features

    Each MS1 spectrum from raw-file will create a feature file -
    we need to load and align them to get unique and representative features
    :param feature_xml_lis:
    :param consensus_map_out_path:
    :return: consensus_map, consensus_map_out_path, measurement_names
    """
    # do consensus map normalization and export -
    # can't hack normalization together from lack of example usage and poor signature
    #   - no normalization implemented

    # openms won't deal with posix paths - wants to have strings instead
    # need to make sure it get's those
    # let's sort them to make sure feature matrix is also sorted
    feature_xml_lis = sorted([str(fx) for fx in feature_xml_lis])

    num_features_list = []
    for current_feature_xml_path in feature_xml_lis:
        # load features into FeatureMaps
        cm = oms.FeatureMap()  # current_map
        oms.FeatureXMLFile().load(current_feature_xml_path, cm)
        # list_functions(current_map, prefix="")
        num_features_list.append(cm.size())
        del cm

    # should choose the feature file / experiment with most features as reference
    max_index = np.argmax(num_features_list)
    reference_map_path = feature_xml_lis[max_index]

    default_max_num_peaks_considered = 1000
    default_max_scaling_value = 10.0
    aligned_paths = []
    for current_feature_xml_path in feature_xml_lis:
        # load features into FeatureMaps
        reference_map = oms.FeatureMap()  # pairwise alignment - so need master map -
        oms.FeatureXMLFile().load(reference_map_path, reference_map)

        current_map = oms.FeatureMap()
        oms.FeatureXMLFile().load(current_feature_xml_path, current_map)

        # create a transformation description required as init for aligner
        transformation_description = oms.TransformationDescription()

        # adjust max scaling parameter otherwise leads to error when running with algae samples
        # adjust max num peaks to 2k - also would leads to error when running with algae samples

        aligner = oms.MapAlignmentAlgorithmPoseClustering()
        aligner_params = aligner.getParameters()

        # print(aligner_params.asDict().keys())
        max_scaling_key = b'superimposer:max_scaling'
        # aligner_params.getEntry(max_scaling_key)
        aligner_params.setValue(max_scaling_key, default_max_scaling_value)

        max_num_peaks_key = b'max_num_peaks_considered'
        # aligner_params.getEntry(max_num_peaks_key)
        aligner_params.setValue(max_num_peaks_key, default_max_num_peaks_considered)  # default = 1000
        # need higher default for algae

        # decrease runtime by removing weak signals
        # print(aligner_params.asDict())
        num_used_points_key = b'superimposer:num_used_points'
        # aligner_params.getEntry(num_used_points_key)
        aligner_params.setValue(num_used_points_key, 1000)  # half the default parameter, speed up alignment

        aligner.setParameters(aligner_params)

        aligner.setReference(reference_map)

        try:
            # run alignment
            aligner.align(current_map, transformation_description)
        except RuntimeError as re:
            if 'max_num_peaks_considered' in str(re):
                # retry with higher threshold - required for algae dataset
                default_max_num_peaks_considered = 15000  # 15 fold - makes it a lot slower but less error prone
                aligner_params.setValue(max_num_peaks_key, default_max_num_peaks_considered)
                default_max_scaling_value = 20.0  # need to increase to 20
                aligner_params.setValue(max_scaling_key, default_max_scaling_value)

                # max shift could also be off - issue for ckd dataset
                default_max_shift_value = 2000.0  # need to increase from 1000 to 2000
                max_shift_key = b'superimposer:max_shift'
                aligner_params.setValue(max_shift_key, default_max_shift_value)

                print(f"Encountered GC/MS Clustering issue - setting 'max_num_peaks_considered' to {default_max_num_peaks_considered}, 'superimposer:max_scaling' to {default_max_scaling_value} and 'superimposer:max_shift' to {default_max_shift_value}")
                aligner.setParameters(aligner_params)
                aligner.setReference(reference_map)
                aligner.align(current_map, transformation_description)


        current_map.updateRanges()
        reference_map.updateRanges()

        # update feature XML files - both reference and current
        updated_current_map_path = default_store_aligned_feature_xml(current_map, current_feature_xml_path)
        updated_reference_path = default_store_aligned_feature_xml(reference_map, reference_map_path)
        reference_map_path = updated_reference_path

        aligned_paths.append(updated_current_map_path)

    # also replace here with new reference we updated the reference map to
    aligned_paths[max_index] = reference_map_path

    #   link/group them across features to create consensus map

    grouper = oms.FeatureGroupingAlgorithmUnlabeled()
    # leave parameters default

    # according to openms documentation:
    #   b) Call "setReference", "addToGroup" (n times), "getResultMap" in that order.

    for i, current_feature_map_path in enumerate(aligned_paths):
        current_map = oms.FeatureMap()
        oms.FeatureXMLFile().load(current_feature_map_path, current_map)

        if not i:
            # first iteration - use as reference
            grouper.setReference(i, current_map)

        else:
            grouper.addToGroup(i, current_map)

    # get consensus map
    consensus_map = grouper.getResultMap()

    # consensus map requires some mapping between ids and filenames - otherwise will complain
    print(f"Mapping aligned results back to class labels")
    class_label_fns = list(class_label_dict.keys())
    fds = {i: oms.ColumnHeader() for i, _ in enumerate(aligned_paths)}
    measurement_names = []
    for i, aligned_path in enumerate(aligned_paths):
        # fds[i].filename = b"file0"
        current_fn = f"{str(Path(aligned_path).stem)}{str(Path(aligned_path).suffix)}"

        # this is where we need to replace the feature_xml filenames with the ones from class_labels
        if class_label_dict:
            # could do longest substring match with each of the fns in class_label dict to find matching filename
            #   django will rename duplicate filenames instead of overwriting
            # or we expect both featureXML input and class_label_dict to be ordered - which they should be when using the getter
            fds[i].filename = class_label_fns[i]

        else:
            fds[i].filename = current_fn.encode("UTF8")  # needs bytestring representation

        measurement_names.append(current_fn)

    consensus_map.setColumnHeaders(fds)

    #  cleanup aligned_feature_xmls - can be >30mb per file - so better remove them
    for ap in aligned_paths:
        os.remove(ap)

    #   do consensus map normalization and export to consensus files
    # using median normalization, also available are Quantile and "robust regression"
    normalizer = oms.ConsensusMapNormalizerAlgorithmMedian()

    # ConsensusMapNormalizerAlgorithmMedian
    # signature of class is more than incomplete ... *args **kwargs for required parameters is not the best implementation choice...
    # but gives TypeError requiring int when calling with
    # normalizer.normalizeMaps(consensus_map, "NM_SCALE", "", "") #
    """
    normalizer.normalizeMaps(map, method, acc_filter, desc_filter)
    map	ConsensusMap
    method	whether to use scaling or shifting to same median 
    acc_filter	string describing the regular expression for filtering accessions
    desc_filter	string describing the regular expression for filtering descriptions 
    """
    """
        method: probably 0 / 1 - referenced as Enumerator in OpenMS documentation
        from shell output can deduce normalization methods are
        0: NM_SCALE 	scale to same median using division/multiplication  
        1: NM_SHIFT 	shift using subtraction/addition
    """
    normalizer.normalizeMaps(consensus_map, 0, "", "")

    # don't export if not required - requires more file management
    # now export
    if consensus_map_out_path:
        oms.ConsensusXMLFile().store(str(consensus_map_out_path), consensus_map)

    return consensus_map, measurement_names


def convert_consensus_map_to_feature_matrix(consensus_map, measurement_names, rounding_precision=0):
    """
    Create a `pandas.DataFrame` from `pyopenms.ConsensusMap`. Col-index holds PeakIds and row-index measurement names
        without suffix
    :param consensus_map: `pyopenms.ConsensusMap`
    :param measurement_names: measurement names in same order as referenced in consensus map index
    :return:
    """
    all_feature_records = []
    peak_ids = []
    for consensus_feature in consensus_map:
        # consensus_feature.computeConsensus()
        peak_id = generate_consensus_peak_id(consensus_feature, rounding_precision=rounding_precision)
        # print("ConsensusFeature", peak_id)

        # consensus feature clusters are retrieved - but don't have unique or persistent ids - concat f"{rt}_{mz}_{charge}" as feature id
        # consensus_map.getColumnHeaders()[0]

        single_col_dict = {}
        # now get intensity for each of the files
        # lets just assume that they are always in order
        for single_feature in consensus_feature.getFeatureList():
            # print(" -- Feature", single_feature.getMapIndex(), round(single_feature.getIntensity(), 3))
            single_col_dict[single_feature.getMapIndex()] = round(single_feature.getIntensity(), 3)

        all_feature_records.append((peak_id, single_col_dict))
        peak_ids.append(peak_id)

    # add all to pandas dataframe
    # sort by peak_id
    sorted_feature_records = sorted(all_feature_records, key=lambda t: t[0])
    num_rows = len(measurement_names)
    feature_df = pd.DataFrame()
    for (peak_id, col_dict) in sorted_feature_records:
        numpy_col = np.zeros(num_rows, dtype=float)
        for index, intensity in col_dict.items():
            numpy_col[index] = intensity
        feature_df[peak_id] = numpy_col

    # index_names = [mn.split(".mz")[0] + ".mzML" for mn in measurement_names]
    # shouldn't assume only mzml - cn also be mzxml!
    # index_names = [str(mn).split(".mz")[0] for mn in measurement_names]
    # dont split - use measurement names passed
    index_names = measurement_names
    feature_df.index = index_names
    return feature_df


def default_store_feature_xml(feature_map, parameters):
    """
    Store feature xml file in fedault location with default name - standaradized manner
    :param feature_map:
    :param parameters:
    :return: updated parameters with 'feature_storage' set to path
    """
    fh = oms.FeatureXMLFile()
    feature_xml_outname = f"{parameters['input_filename']}{get_default_feature_xml_storage_suffix()}"
    fh.store(feature_xml_outname, feature_map)
    parameters['feature_storage'] = feature_xml_outname
    return parameters

def default_store_aligned_feature_xml(feature_map, original_path):
    """
    Store feature xml file in default location with default aligned name
    :param feature_map:
    :param parameters:
    :return: updated parameters with 'feature_storage' set to path
    """
    fh = oms.FeatureXMLFile()
    storage_suffix = get_default_feature_xml_storage_suffix('aligned')
    if str(original_path).endswith(storage_suffix):
        # overwrite - probably writing the reference feature map
        feature_xml_outname = original_path
    else:
        feature_xml_outname = f"{Path(original_path).parent}/{Path(original_path).stem}{storage_suffix}"

    fh.store(feature_xml_outname, feature_map)
    return feature_xml_outname


def get_default_feature_xml_storage_suffix(suffix_prefix=""):
    """"
    :return: Default storage suffix for feature xml files
    """
    if suffix_prefix:
        return f"_{suffix_prefix}_output.featureXML"
    else:
        return f"_output.featureXML"

def list_functions(obj, prefix="get"):
    for fun in dir(obj):
        if prefix:
            if str(fun).startswith(prefix):
                print(fun)
        else:
            if not str(fun).startswith("__"):
                print(fun)


def denoise_sgf_gcms(ms_experiment):
    # chroms = ms_experiment
    sg = oms.SavitzkyGolayFilter()
    sg.filterExperiment(ms_experiment)
    # out_path = f"{path_to_mzml}.filter.mzML"
    # # TODO they changed the order of arguments in the store method in v2.5 - be aware!
    # oms.MzMLFile().store(out_path, chroms)
    return ms_experiment


def generate_consensus_peak_id(consensus_feature, rounding_precision=0):
    """
    Generate the consonsus peak id by concatenating rt_mz_charge - trailing zeros for sortability to reproduce same ordering
    :param consensusPeak:
    :return:
    """
    rp = rounding_precision
    if rp:
        return f"Peak_{round(consensus_feature.getRT(),rp):05}_{round(consensus_feature.getMZ(),rp):05}_{int(consensus_feature.getCharge())}"
    else:  # remove the .0 from the peak id - as we're rounding the digits
        rt = int(round(consensus_feature.getRT(), rp))
        mz = int(round(consensus_feature.getMZ(), rp))
        ch = int(consensus_feature.getCharge())
        return f"Peak_{rt}_{mz}_{ch}"


def filter_mzml_or_mzxml_filenames(dir, filelis=[]):
    """
    Return all mz and mzxml filenames, if filelis supplied (eg from namelist of zip - return suset of them
    :param dir:
    :param filelis:
    :return:
    """
    if not filelis:
        if not str(dir).endswith("/"):
            dir = str(dir) + "/"

        paths_mzml = glob.glob(dir + "*.mzML")
        paths_mzml2 = glob.glob(dir + "*.mzml")

        paths_mzxml = glob.glob(dir + "*.mzXML")
        paths_mzxml2 = glob.glob(dir + "*.mzxml")

        all_paths = []
        all_paths.extend(paths_mzml)
        all_paths.extend(paths_mzml2)
        all_paths.extend(paths_mzxml)
        all_paths.extend(paths_mzxml2)

    else:
        all_paths = []
        for fn in filelis:
            if str(fn).lower().endswith(".mzml") or str(fn).lower().endswith(".mzxml"):
                all_paths.append(fn)

    return all_paths


def detect_raw_peaks_wavelet_single_core(fn_in):
    """
    Apply wavelet transformation and pick peaks on raw MS1 spectra, expecting
    :param parameters:
    :return: filename of feature xml file
    """
    current_params = dict()
    current_params['input_filename'] = fn_in
    fn_out = run_raw_wavelet_sample(current_params)['feature_storage']
    print(f"Saved features to {fn_out}")
    return fn_out


def run_centroid_sample(parameters):
    """
    Run centroid gcms detection. Expects 'input_filename' to be set. Use 'feature_storage' to access result fn
    :returns dict('feature_storage': 'fn')
    :param parameters:
    :return: dict()
    """
    ms_experiment = oms.MSExperiment()
    ms_experiment = load_ms_file(ms_experiment=ms_experiment, path_to_file=parameters['input_filename'])
    return detect_peaks_gcms_centroid(ms_experiment=ms_experiment, parameters=parameters)


def run_raw_wavelet_sample(parameters):
    """
    Run wavelet based gcms detection. Expects 'input_filename' to be set. Use 'feature_storage' to access result fn
    :returns dict('feature_storage': 'fn')
    :param parameters:
    :return: dict()
    """
    ms_experiment = oms.MSExperiment()
    ms_experiment = load_ms_file(ms_experiment=ms_experiment, path_to_file=parameters['input_filename'])
    # ms_experiment = denoise_sgf_gcms(ms_experiment)
    return detect_peaks_gcms_isotopewavelet(ms_experiment=ms_experiment, parameters=parameters)
