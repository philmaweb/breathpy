from collections import OrderedDict
import glob
from multiprocessing import Pool
import numpy as np
import os
from pathlib import Path
import pyopenms as oms
from shutil import rmtree
import tempfile

import joblib

from .BreathCore import MccImsAnalysis, GCMSAnalysis, PredictionModel, construct_default_processing_evaluation_steps, construct_default_parameters

from .GCMSTools import (
    align_feature_xmls,
    convert_consensus_map_to_feature_matrix,
    GCMSPeakDetectionMethod,
    list_functions,
    filter_mzml_or_mzxml_filenames,
    run_centroid_sample,
    filter_feature_xmls,
)
from .ProcessingMethods import GCMSAlignmentMethod
from ..view.BreathVisualizations import HeatmapPlot, ClusterPlot, MaskPlot, RocCurvePlot, BoxPlot, TreePlot


def run_alignment_test(feature_xml_lis):
    """
    Uses all FeatureXML to create peak alignment result
    :param parameters:
    :return:
    """
    # pass intermediary consensusXML filename
    # TODO do we need to store a real copy of the consensus xml and intermediaries - clean up after?
    consensus_outfile_name = f"{Path(feature_xml_lis[0]).parent}/test.consensusXML"
    consensus_map, measurement_names = align_feature_xmls(feature_xml_lis, consensus_outfile_name)
    feature_df = convert_consensus_map_to_feature_matrix(consensus_map, measurement_names)
    return feature_df


def run_algae_platform():
    """
    Run algae example from start to finish using BreathCore implementations and classes
    :return:
    """
    data_dir = "/home/philipp/dev/breathpy/data"
    algae_dir = f"{data_dir}/algae/"
    algae_files, class_label_dict = prepare_algae_sample_filenames(algae_dir, ['dark', 'replete', 'light'])
    # gcms_a = GCMSAnalysis(measurements=algae_files, preprocessing_steps=[GCMSPeakDetectionMethod.ISOTOPEWAVELET], dataset_name="algae_limited", dir_level=".", preprocessing_parameters={GCMSPeakDetectionMethod.ISOTOPEWAVELET: {"hr_data": True}}, performance_measure_parameters={})
    gcms_a = GCMSAnalysis(measurements=algae_files, preprocessing_steps=[GCMSPeakDetectionMethod.ISOTOPEWAVELET], dataset_name="algae", dir_level=".", preprocessing_parameters={GCMSPeakDetectionMethod.ISOTOPEWAVELET: {"hr_data": True}}, performance_measure_parameters={})
    gcms_a.set_class_label_dict(class_label_dict)
    gcms_a.detect_peaks_parallel(num_cores=4)
    # gcms_a.detect_peaks()
    # feature alignment fails - some nan error during pose map alignment
    feature_xml_lis = filter_feature_xmls(algae_dir)
    consensus_map, measurement_names = align_feature_xmls(feature_xml_lis, "test.out")

    # TODO use restore feature - dont want to keep rerunning peak detection to get to feature df
    # feature_df = gcms_a.align_peaks(GCMSAlignmentMethod.POSE_FEATURE_GROUPER)
    # print(feature_df)

def run_tiny_samples_parallel(num_cores=4, sample_dir="/home/philipp/Desktop/gcms_data/sample"):
    sample_dir = Path(sample_dir)

    # mini_sample_path_to_mzml = Path(str(urine_dir) + "/" + "tiny.pwiz.1.1.mzML")
    sample_path_to_mzml = sample_dir/"FeatureFinderCentroided_1_input.mzML"

    # use same tiny file several times
    example_fns = []
    example_params = []
    for i in range(3):
        current_sample_path = Path(f"{sample_dir}/{sample_path_to_mzml.stem}_{i}.mzML")
        example_fns.append(current_sample_path)
        example_params.append({'input_filename': current_sample_path})

    with Pool(num_cores) as p:
        feature_xml_dicts = p.map(run_centroid_sample, example_params)

    feature_xml_lis = []
    # list will not be the same order as before - so need to sort before aligning
    for params in feature_xml_dicts:
        feature_xml_fn = params['feature_storage']
        feature_xml_lis.append(feature_xml_fn)
    consensus_map, measurement_names = align_feature_xmls(feature_xml_lis, "test.out")
    feature_df = convert_consensus_map_to_feature_matrix(consensus_map, example_fns)
    print(feature_df)
    return feature_df


def prepare_algae_sample_filenames(algae_dir, limit_to_classes=["n-limited"]):
    """
    Return paths to algae files
    :param algae_dir:
    :return:
    """
    algae_files = filter_mzml_or_mzxml_filenames(algae_dir)

    # get class label dict from zip / archive

    label_dict_path = MccImsAnalysis.guess_class_label_extension(algae_dir)
    label_dict = MccImsAnalysis.parse_class_labels(label_dict_path)

    if not limit_to_classes:
        return algae_files, label_dict

    # apply filtering by class label
    else:
        filtered_label_dict = OrderedDict()
        sorted_algae_files = sorted(algae_files)
        # now also need to filter sorted algae files


        for fn, label in label_dict.items():
            if label in limit_to_classes:
                filtered_label_dict[fn] = label

        filtered_sorted_algae_files = []
        for fn in sorted_algae_files:
            if str(Path(fn).stem) + str(Path(fn).suffix) in filtered_label_dict:
                filtered_sorted_algae_files.append(fn)

        return filtered_sorted_algae_files, label_dict

# use https://pyopenms.readthedocs.io/en/latest/datastructures_quant.html
#   need to differentiate between centroided and raw data - if centroided cant apply savitzky golay
#  user needs to tell us whether centroided data or not -
#  is encoded in mzML-file file description tag -
#  (cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum") -
#   - but can't find documentation to access cvParams to check - there is CVReference and CVMapping class but how to access / initialize these...


def run_gcms_platform_multicore(sample_dir, preprocessing_params={}, evaluation_parms={}, num_cores=4):
    """
    Run platform with default parameters unless specified -
        will look for existing results in results/data/sample_dir and if sample_dir starts with 'train_' if will use the
        same path with 'test_' for prediction
    :param sample_dir:
    :return:
    """
    dataset_name = str(Path(sample_dir).stem)
    print(f"Starting analysis of {dataset_name}")
    plot_params, file_params = construct_default_parameters(dataset_name, folder_name=dataset_name, make_plots=True)

    raw_files, class_label_dict = GCMSAnalysis.prepare_raw_filenames_class_labels(sample_dir)
    gcms_a = GCMSAnalysis(measurements=raw_files,
                          preprocessing_steps=preprocessing_params.keys(),
                          dataset_name=dataset_name, dir_level=".",
                          preprocessing_parameters=preprocessing_params,
                          performance_measure_parameters=evaluation_parms)
    gcms_a.set_class_label_dict(class_label_dict)

    feature_xml_lis = filter_feature_xmls(sample_dir)
    if not feature_xml_lis:
        gcms_a.detect_peaks_parallel(num_cores=num_cores)
        # gcms_a.detect_peaks()
    feature_xml_lis = filter_feature_xmls(sample_dir)

    if len(class_label_dict) != len(feature_xml_lis):
        raise ValueError(f"Class labels don't match feature XMLs {len(class_label_dict)} != {len(feature_xml_lis)}.")
    # print(feature_xml_lis)

    # cross validation and plots
    # import peak detection results - but don't use aligned ones if already present - will lead to problems downstream
    gcms_a.import_peak_detection_results(feature_xml_lis)

    # Next: create Feature matrix
    # import feature matrix if already existent - otherwise do alignment
    result_data_dir = f"{gcms_a.dir_level}/results/data/{gcms_a.dataset_name}/"
    try:
        # can take several minutes to load - some alignment dfs are hundred of MBs in size
        fm = gcms_a.import_alignment_result(result_data_dir, gcms_a.peak_detection_steps[0].name)
    except FileNotFoundError:
        fm = gcms_a.align_peaks(GCMSAlignmentMethod.POSE_FEATURE_GROUPER)
    # consensus_map, measurement_names = align_feature_xmls(feature_xml_lis, f"{dataset_name}.out", class_label_dict)

    ims_analysis = gcms_a.prepare_custom_fm_approach(fm)

    # expects a FloatPeakAlignmentResult - which we never have coming from GCMS
    # ims_analysis.reduce_features(ims_analysis.AVAILABLE_FEATURE_REDUCTION_METHODS)
    ims_analysis.remove_redundant_features_fm()
    # reduced_fm_dict = ims_analysis.analysis_result.remove_redundant_features_fm(ims_analysis.analysis_result)
    ims_analysis.evaluate_performance(exising_analysis_result=ims_analysis.analysis_result)
    # if len(set(ims_analysis.analysis_result.class_labels)) == 2:
    #     RocCurvePlot.ROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)
    # else:
    #     RocCurvePlot.MultiClassROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)
    # BoxPlot.BoxPlotBestFeature(ims_analysis.analysis_result, plot_parameters=plot_params)
    if plot_params['make_plots']:
        TreePlot.DecisionTrees(ims_analysis.analysis_result, plot_parameters=plot_params)

    # continue with prediction and preprocessing of test set
    tmp = os.path.join(tempfile.gettempdir(), '.breath/{}'.format(hash(os.times())))
    os.makedirs(tmp)

    dataset_name = str(Path(sample_dir).stem)
    if dataset_name.startswith("train_"):
        # actually using sample_dir which is at a different location
        test_dir = str(sample_dir).replace("train_", "test_")
        test_dataset_name = dataset_name.replace("train_", "test_")
        test_result_data_dir = f"{gcms_a.dir_level}/results/data/{gcms_a.dataset_name}/".replace("train_", "test_")
    else:
        test_dir = sample_dir
        test_dataset_name = dataset_name
        test_result_data_dir = f"{gcms_a.dir_level}/results/data/{gcms_a.dataset_name}/"

    #  handle for GCMS files
    predictor_path = tmp + "/pred_model.sav"  # export and reload for usage later on
    ims_analysis.analysis_result.export_prediction_models(path_to_save=predictor_path)
    predictors = joblib.load(predictor_path)

    # clean up tempdir
    rmtree(tmp, ignore_errors=True)

    test_files, test_class_label_dict = GCMSAnalysis.prepare_raw_filenames_class_labels(test_dir)

    test_gcms_a = GCMSAnalysis(measurements=test_files,
                          preprocessing_steps=preprocessing_params.keys(),
                          dataset_name=test_dataset_name, dir_level=".",
                          preprocessing_parameters=preprocessing_params,
                          performance_measure_parameters=evaluation_parms)
    test_gcms_a.set_class_label_dict(test_class_label_dict)

    # convert to feature matrix - or load if available - fallback to detection import + alignment, fallback to full pipeline

    test_feature_xml_lis = filter_feature_xmls(test_dir)
    if not test_feature_xml_lis:
        gcms_a.detect_peaks_parallel(num_cores=4)
    test_feature_xml_lis = filter_feature_xmls(test_dir)

    if len(test_class_label_dict) != len(test_feature_xml_lis):
        raise ValueError(f"Class labels don't match feature XMLs {len(test_class_label_dict)} != {len(test_feature_xml_lis)}.")
    print(test_feature_xml_lis)

    # import peak detection results - but don't use aligned ones if already present - will lead to problems downstream
    gcms_a.import_peak_detection_results(test_feature_xml_lis)

    # Next: create Feature matrix

    # import feature matrix if already existent - otherwise do alignment
    try:
        # can take several minutes to load - some alignment dfs are hundred of MBs in size
        test_fm = gcms_a.import_alignment_result(test_result_data_dir, gcms_a.peak_detection_steps[0].name)

    except FileNotFoundError:
        test_fm = gcms_a.align_peaks(GCMSAlignmentMethod.POSE_FEATURE_GROUPER)
    # consensus_map, measurement_names = align_feature_xmls(feature_xml_lis, f"{dataset_name}.out", class_label_dict)

    test_ims_analysis = gcms_a.prepare_custom_fm_approach(test_fm)

    # make sure features match
    train_column_names = ims_analysis.analysis_result.trainings_matrix[gcms_a.peak_detection_steps[0].name].columns

    trainings_feature_names_by_pdmn = {gcms_a.peak_detection_steps[0].name: train_column_names}

    # make test_fm match fm
    # test_fm_dict = {gcms_a.peak_detection_steps[0].name: test_fm}
    test_fm_dict = test_ims_analysis.analysis_result.feature_matrix
    df_dict_matching_training = PredictionModel.reconstruct_remove_features(test_fm_dict, trainings_feature_names_by_pdmn)

    predictionModel = PredictionModel(
        preprocessing_params=preprocessing_params,
        evaluation_params=ims_analysis.performance_measure_parameter_dict,
        scipy_predictor_by_pdm=predictors,
        feature_names_by_pdm=ims_analysis.analysis_result.feature_names_by_pdm,)  # not sure whether important to use train / test matrix

    prediction, test_matrix_by_pdm = predictionModel.predict_from_feature_matrix(pdms=[GCMSPeakDetectionMethod.ISOTOPEWAVELET], fm_dict_matching_training=df_dict_matching_training)
    # is always sorted
    prediction_holder = []
    test_class_labels = np.unique(list(test_class_label_dict.values()))
    for pdm, prediction_index in prediction.items():
        predicted_labels = {}
        for p, test_name in zip(prediction_index, test_class_label_dict.keys()):
            predicted_labels[test_name] = test_class_labels[p]
        correct = dict()
        false = dict()
        for fn, predicted_label in predicted_labels.items():
            if predicted_label == test_class_label_dict[fn]:
                correct[fn] = predicted_label
            else:
                false[fn] = predicted_label

        print("resulting_labels for {} are: {}".format(pdm.name, predicted_labels))
        print("Falsely classified: {}".format(false))
        print("That's {} correct vs {} false".format(len(correct.keys()), len(false.keys())))
        prediction_holder.append((predicted_labels, correct, false))
    return prediction_holder


# USER inputs whether centroided or not
if __name__ == "__main__":
    params = {}
    sample_dir = Path("/home/philipp/Desktop/gcms_data/sample")

    sample_path_to_mzml = Path(sample_dir/"FeatureFinderCentroided_1_input.mzML")
    params['input_filename'] = sample_path_to_mzml

    # centroid_sample_path_to_mzml = Path(str(urine_dir) + "/" + "FeatureFinderCentroided_1_input.mzML")
    # params['input_filename'] = centroid_sample_path_to_mzml
    # use same tiny file several times
    # feature_xml_lis = []
    # for i in range(3):
    #     current_sample_path = Path(f"{sample_dir}/{sample_path_to_mzml.stem}_{i}.mzML")
    #     params['input_filename'] = current_sample_path
    #     params = run_centroid_sample(params)
    #     feature_xml_fn = params['feature_storage']
    #     feature_xml_lis.append(feature_xml_fn)
    # feature_df = run_alignment_test(feature_xml_lis)

    # what if there are several experiments in single mzml file? they get huge - several GB - need to rethink indexing by source file
    #   sourceFile / sourceFiles attribute in mzML
    # according to https://github.com/OpenMS/OpenMS/issues/1655 abuse of file format - one run is one mzML file -> not supported

    # save feature matrix
    run_tiny_samples_parallel(3)
    run_algae_platform()