import glob
from pathlib import Path

import joblib

import tempfile
import os
from shutil import rmtree

import numpy as np

import pandas
from .ProcessingMethods import (
    PeakDetectionMethod,
    ExternalPeakDetectionMethod,
    NormalizationMethod,
    PeakAlignmentMethod,
    DenoisingMethod,
    PerformanceMeasure,
    FeatureReductionMethod,
)
from .BreathCore import (MccImsMeasurement,
                              MccImsAnalysis,
                              AnalysisResult,
                              PredictionModel,
                              construct_default_parameters,
                              construct_default_processing_evaluation_steps,
                              get_breath_analysis_dir,
                              )
from ..view.BreathVisualizations import HeatmapPlot, ClusterPlot, MaskPlot, RocCurvePlot, VennDiagram, BoxPlot, TreePlot
from ..tools.tools import file_limit_stratify_selection_by_label

def test_start_to_end_pipeline(plot_params, file_params, preprocessing_steps, evaluation_params_dict, number_of_files_limit=-1, specific_classes=[], stop_after_alignment=False):
    """
    Run pipeline with raw MCC/IMS measurements
    :param plot_params:
    :param file_params:
    :param preprocessing_steps:
    :param evaluation_params_dict:
    :param number_of_files_limit:
    :param specific_classes:
    :param stop_after_alignment: `bool` - stop after peak alignment
    :return:
    """
    full_path_to_in_files = glob.glob(file_params['folder_path'] + "*_ims.csv")

    label_dict_path = MccImsAnalysis.guess_class_label_extension(file_params['folder_path'])
    label_dict = MccImsAnalysis.parse_class_labels(label_dict_path)

    if PeakDetectionMethod.VISUALNOWLAYER in preprocessing_steps:
        visualnow_layer_path = [filename for filename in glob.glob(file_params['folder_path'] + "*") if
                        (str.endswith(filename, "layer.csv") or str.endswith(filename, "layer.xls"))][0]
    else:
        visualnow_layer_path = ""

    in_file_names = [fp.rsplit("/", maxsplit=1)[-1] for fp in full_path_to_in_files]

    # filter input flenames by assigned class and stratify by class
    in_file_names = file_limit_stratify_selection_by_label(in_file_names, label_dict, labels_to_keep=specific_classes, file_limit=number_of_files_limit)

    # also limit class labels to used files - otherwise feature reduction will fail check
    if number_of_files_limit !=-1:
        new_class_label_dict = dict()
        for fn in in_file_names:
            new_class_label_dict[fn] = label_dict[fn]
        label_dict = new_class_label_dict

    # check if output directory already exists
    if not Path(file_params['out_dir']).exists():
        # create directory if it doesnt exist already
        Path(file_params['out_dir']).mkdir(parents=True, exist_ok=True)

    # check if already peaxed, if yes, then read in peaxed files
    outfile_names = [file_params['out_dir'] + fn[:-4] + "_out.csv" for fn in in_file_names]
    # are_all_files_already_peaxed = all([Path(on).exists() for on in outfile_names])

    print("Extracting CSV files {}".format(in_file_names))
    # Use parse raw measurement instead of extracting from zip everytime
    # my_csv_df = extract_csvs_from_zip(raw_zip_archive, verbose=True)
    my_ims_measurements = [MccImsMeasurement(f"{file_params['folder_path']}{i_fn}") for i_fn in in_file_names]

    # my_ims_measurements = [MccImsMeasurement(i_fn) for i_fn in full_path_to_in_files]
    print("Finished Extracting CSV files.\n")
    print(my_ims_measurements)

    peax_binary_path = f"{get_breath_analysis_dir()}/bin/peax1.0-LinuxX64/peax"
    ims_analysis = MccImsAnalysis(my_ims_measurements, preprocessing_steps, outfile_names,
                                  performance_measure_parameters=evaluation_params_dict,
                                  class_label_file=label_dict_path,
                                  dir_level=file_params['dir_level'],
                                  dataset_name=file_params['folder_name'],
                                  visualnow_layer_file=visualnow_layer_path,
                                  peax_binary_path=peax_binary_path
                                   )

    if number_of_files_limit != -1:
        # also set the class label dict accordingly
        ims_analysis.set_class_label_dict(label_dict)


    # plot_raw_and_processed(ims_analysis, plot_parameters)
    ims_analysis.preprocess_multicore()

    # if plot_params['make_plots']:
    #     for m in ims_analysis.measurements:
    #         HeatmapPlot.FastIntensityMatrix(m, plot_parameters)


    # Make Average classwise Plot
    # HeatmapPlot.ClasswiseHeatmaps(ims_analysis, plot_parameters)
    ims_analysis.export_results_to_csv(file_params['out_dir'])

    ims_analysis.align_peaks(file_prefix=file_params['file_prefix'])
    if plot_params['make_plots']:
        # make sure folder exists
        venn_dir_path = Path(f"{plot_params['plot_dir']}venn_diagram/")
        venn_dir_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(ims_analysis.comparison_peak_detection_result, pandas.core.frame.DataFrame):
            pass
        elif ims_analysis.comparison_peak_detection_result.shape[0] == 2:
            VennDiagram.Venn2Diagram(ims_analysis.comparison_peak_detection_result, plot_parameters=plot_params)
        elif ims_analysis.comparison_peak_detection_result.shape[0] == 3:
            VennDiagram.Venn3Diagram(ims_analysis.comparison_peak_detection_result, plot_parameters=plot_params)
        elif ims_analysis.comparison_peak_detection_result.shape[0] == 4:
            VennDiagram.Venn4Diagram(ims_analysis.comparison_peak_detection_result, plot_parameters=plot_params)
        else:
            pass

        # orig_prefix = plot_params['plot_prefix']
        # ClusterPlot.OverlayClasswiseAlignment(ims_analysis, plot_parameters=plot_params)
        # plot_params['plot_prefix'] = orig_prefix + "_slow"
        # ClusterPlot.OverlayAlignment(ims_analysis, plot_parameters=plot_params)
        # plot_params['plot_prefix'] = orig_prefix
    if stop_after_alignment:
        return ims_analysis

    if plot_params['make_plots']:
        ClusterPlot.ClusterBasic(ims_analysis, plot_parameters=plot_params)
        # ClusterPlot.ClusterMultiple(ims_analysis, plot_parameters=plot_params)

    ims_analysis.reduce_features(ims_analysis.AVAILABLE_FEATURE_REDUCTION_METHODS)
    ims_analysis.evaluate_performance()

    if plot_params['make_plots']:

        if ims_analysis.is_able_to_cross_validate():
            if len(set(ims_analysis.analysis_result.class_labels)) == 2:
                RocCurvePlot.ROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)
            else:
                RocCurvePlot.MultiClassROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)

        BoxPlot.BoxPlotBestFeature(ims_analysis.analysis_result, plot_parameters=plot_params)
        TreePlot.DecisionTrees(ims_analysis.analysis_result, plot_parameters=plot_params)

    # continue with prediction and preprocessing of test set
    tmp = os.path.join(tempfile.gettempdir(), f'.breath/{hash(os.times())}')
    os.makedirs(tmp)

    dataset_name = file_params['folder_path'].split("/")[-1]
    # some splits don't work - as we have a trailing /
    if not dataset_name:
        dataset_name = file_params['folder_path'].split("/")[-2]
    if dataset_name.startswith("train_"):
        # update dir where files are read from - for test training
        # otherwise can't find measurements - as the class labels don't match the measurement names
        file_params['folder_path'] = file_params['folder_path'].replace("train_", "test_")
        test_dir = file_params['folder_path'].replace("train_", "test_")
    else:
        test_dir = file_params['folder_path']

    test_full_path_to_in_files = glob.glob(file_params['folder_path'] + "*_ims.csv")
    test_in_file_names = [fp.split("/")[-1] for fp in test_full_path_to_in_files]

    test_labels_dict = MccImsAnalysis.parse_class_labels(MccImsAnalysis.guess_class_label_extension(test_dir))
    filtered_stratified_test_names = file_limit_stratify_selection_by_label(test_in_file_names, keys_to_label_dict=test_labels_dict, labels_to_keep=specific_classes, file_limit=number_of_files_limit)
    test_paths = [f"{test_dir}{i_fn}" for i_fn in filtered_stratified_test_names]
    test_measurements = [MccImsMeasurement(tp) for tp in test_paths]

    predictor_path = tmp+"/pred_model.sav"
    ims_analysis.analysis_result.export_prediction_models(path_to_save=predictor_path)
    predictors = joblib.load(predictor_path)

    # clean up tempdir
    rmtree(tmp, ignore_errors=True)

    predictionModel = PredictionModel(
        preprocessing_params={s:{} for s in preprocessing_steps},
                                      evaluation_params=ims_analysis.performance_measure_parameter_dict,
                                      scipy_predictor_by_pdm=predictors,
                                      feature_names_by_pdm=ims_analysis.analysis_result.feature_names_by_pdm,
                                      peax_binary_path=peax_binary_path,
                                      visualnow_layer_file=visualnow_layer_path)

    prediction = predictionModel.predict(test_measurements)
    # is always sorted

    class_labels = np.unique([m.class_label for m in ims_analysis.measurements])
    for pdm, prediction_index in prediction.items():
        predicted_labels = {test_name: class_labels[p] for p, test_name in zip(prediction_index, filtered_stratified_test_names)}
        correct = dict()
        false = dict()
        for fn, predicted_label in predicted_labels.items():
            if predicted_label == test_labels_dict[fn]:
                correct[fn] = predicted_label
            else:
                false[fn] = predicted_label

        print("resulting_labels for {} are: {}".format(pdm.name, predicted_labels))
        print("Falsely classified: {}".format(false))
        print("That's {} correct vs {} false".format(len(correct.keys()), len(false.keys())))
    return ims_analysis


def test_resume_analysis(plot_params, file_params, preprocessing_steps, evaluation_params_dict, preprocessing_params_dict={},number_of_files_limit=-1, specific_classes=(), stop_after_alignment=False):
    """
    Run pipeline with preprocessed / peak detection results from MCC/IMS measurements
    :param plot_params:
    :param file_params:
    :param preprocessing_steps:
    :param preprocessing_params_dict: specifies method parameters, will overwrite default parameters
    :param evaluation_params_dict:
    :param number_of_files_limit:
    :param specific_classes:
    :param stop_after_alignment: `bool` stop after peak alignment
    :return:
    """
    full_path_to_in_files = glob.glob(file_params['folder_path'] + "*_ims.csv")
    in_file_names = [fp.split("/")[-1] for fp in full_path_to_in_files]

    outfile_names = [file_params['out_dir'] + fn[:-4] + "_out.csv" for fn in in_file_names]

    # test for visualnow layer import
    # visual_now_file_path = "/home/philipp/dev/breathpy/data/train_full_candy/train_full_candy_layer.xls"
    peax_binary_path = "{0}bin/peax1.0-LinuxX64/peax".format(file_params['dir_level'])

    label_dict_path = MccImsAnalysis.guess_class_label_extension(file_params['folder_path'])
    label_dict = MccImsAnalysis.parse_class_labels(label_dict_path)

    if PeakDetectionMethod.VISUALNOWLAYER in preprocessing_steps:
        visualnow_layer_path = [filename for filename in glob.glob(file_params['folder_path'] + "*") if
                        (str.endswith(filename, "layer.csv") or str.endswith(filename, "layer.xls"))][0]
    else:
        visualnow_layer_path = ""
    ims_analysis = MccImsAnalysis([], preprocessing_steps, outfile_names,
                                  preprocessing_parameters=preprocessing_params_dict,
                                  performance_measure_parameters=evaluation_params_dict,
                                  dir_level=file_params['dir_level'],
                                  dataset_name=file_params['folder_name'],
                                  class_label_file=label_dict_path,
                                  visualnow_layer_file=visualnow_layer_path,
                                  peax_binary_path=peax_binary_path)

    ims_analysis.import_results_from_csv_dir(file_params['out_dir'], class_label_file=label_dict_path)


    # save peak detection results with peak detection name and _out.csv
    ims_analysis.align_peaks(file_params['file_prefix'])

    if plot_params['make_plots']:
        ClusterPlot.ClusterBasic(ims_analysis, plot_parameters=plot_params)
        # ClusterPlot.ClusterMultiple(ims_analysis, plot_parameters=plot_params)
    if stop_after_alignment:
        return ims_analysis

    ims_analysis.reduce_features(ims_analysis.AVAILABLE_FEATURE_REDUCTION_METHODS)
    ims_analysis.evaluate_performance()
    # get best model
    best_model_name, feature_names, decision_tree_buffer = ims_analysis.get_best_model()

    if plot_params['make_plots']:
        # ClusterPlot.OverlayAlignment(ims_analysis, plot_parameters=plot_params)
        ClusterPlot.OverlayClasswiseAlignment(ims_analysis, plot_parameters=plot_params)
        # ClusterPlot.OverlayBestFeaturesAlignment(ims_analysis, plot_parameters=plot_params)

    # ims_analysis.analysis_result.export_statistics()
    # make sure cross val was run - minimum

    if ims_analysis.is_able_to_cross_validate():
        if len(set(ims_analysis.analysis_result.class_labels)) == 2:
            RocCurvePlot.ROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)
        else:
            RocCurvePlot.MultiClassROCCurve(ims_analysis.analysis_result, plot_parameters=plot_params)

    BoxPlot.BoxPlotBestFeature(ims_analysis.analysis_result, plot_parameters=plot_params)
    TreePlot.DecisionTrees(ims_analysis.analysis_result, plot_parameters=plot_params)

    # continue with prediction and preprocessing of test set
    tmp = os.path.join(tempfile.gettempdir(), '.breath/{}'.format(hash(os.times())))
    os.makedirs(tmp)

    dataset_name = Path(file_params['folder_path']).stem
    # some splits don't work - as we have a trailing /
    if not dataset_name:
        dataset_name = file_params['folder_path'].split("/")[-2]
    if dataset_name.startswith("train_"):
        # update dir where files are read from - for test training
        # otherwise can't find measurements - as the class labels don't match the measurement names
        file_params['folder_path'] = file_params['folder_path'].replace("train_", "test_")
        test_dir = file_params['folder_path'].replace("train_", "test_")
    else:
        test_dir = file_params['folder_path']

    test_full_path_to_in_files = glob.glob(file_params['folder_path'] + "*_ims.csv")
    test_in_file_names = [fp.split("/")[-1] for fp in test_full_path_to_in_files]

    test_labels_dict = MccImsAnalysis.parse_class_labels(MccImsAnalysis.guess_class_label_extension(test_dir))
    filtered_stratified_test_names = file_limit_stratify_selection_by_label(test_in_file_names, keys_to_label_dict=test_labels_dict, labels_to_keep=specific_classes, file_limit=number_of_files_limit)
    test_paths = [f"{test_dir}{i_fn}" for i_fn in filtered_stratified_test_names]


    predictor_path = tmp+"/pred_model.sav"
    ims_analysis.analysis_result.export_prediction_models(path_to_save=predictor_path)
    predictors = joblib.load(predictor_path)

    # clean up tempdir
    rmtree(tmp, ignore_errors=True)

    predictionModel = PredictionModel(
        preprocessing_params={s:preprocessing_params_dict.get(s, {}) for s in preprocessing_steps},
                                      evaluation_params=ims_analysis.performance_measure_parameter_dict,
                                      scipy_predictor_by_pdm=predictors,
                                      feature_names_by_pdm=ims_analysis.analysis_result.feature_names_by_pdm,
                                      peax_binary_path=peax_binary_path,
                                      visualnow_layer_file=visualnow_layer_path)

    # pass test dir as result - if not available fall back to peak detection
    test_result_dir = file_params['out_dir'].replace("train_", "test_")

    try:
        # try import first
        prediction, test_matrix_by_pdm = predictionModel.predict_from_peak_detection_results(test_result_dir, class_label_file_name=MccImsAnalysis.guess_class_label_extension(test_dir))
    except ValueError:
        # otherwise fall back
        test_measurements = [MccImsMeasurement(tp) for tp in test_paths]
        prediction = predictionModel.predict(test_measurements)

    # is always sorted - and train should always contain all class labels - just needed for finding
    #   the correct labels from prediction = indices of predicted class
    class_labels = np.unique(list(ims_analysis.class_label_dict.values()))

    for pdm, prediction_index in prediction.items():
        if filtered_stratified_test_names:
            predicted_labels = {test_name: class_labels[p] for p, test_name in zip(prediction_index, filtered_stratified_test_names)}
        else:
            predicted_labels = {test_name: class_labels[p] for p, test_name in
                                zip(prediction_index, test_matrix_by_pdm[pdm].index.values)}
        correct = dict()
        false = dict()
        for fn, predicted_label in predicted_labels.items():
            if predicted_label == test_labels_dict[fn]:
                correct[fn] = predicted_label
            else:
                false[fn] = predicted_label

        print("resulting_labels for {} are: {}".format(pdm.name, predicted_labels))
        print("Falsely classified: {}".format(false))
        print("That's {} correct vs {} false".format(len(correct.keys()), len(false.keys())))

    return ims_analysis, prediction



def run_default(set_name, make_plots=False, limit_to_pdm=[]):
    plot_parameters, file_parameters = construct_default_parameters(set_name, set_name, make_plots=make_plots,
                                                                    execution_dir_level='one')

    preprocessing_steps, evaluation_params_dict = construct_default_processing_evaluation_steps()
    if limit_to_pdm:
        # remove pdms that are not in pdm
        filtered = [m for m in preprocessing_steps if not (isinstance(m ,(PeakDetectionMethod, ExternalPeakDetectionMethod)))]
        filtered.extend(limit_to_pdm)
        preprocessing_steps = filtered


    test_start_to_end_pipeline(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict,
                               number_of_files_limit=0)
    return file_parameters, preprocessing_steps

if __name__ == '__main__':
    # file_prefix = 'train_mouthwash'; folder_name = "train_mouthwash"
    # file_prefix = 'train_COPD'; folder_name = "train_COPD"
    # file_prefix = 'train_full_candy'; folder_name = "train_full_candy"
    # file_prefix = 'test_full_candy'; folder_name = "test_full_candy"
    file_prefix = 'small_candy_anon'; folder_name = "small_candy_anon"

    # edit resultsdir to match folder structure
    plot_parameters, file_parameters = construct_default_parameters(file_prefix, folder_name, make_plots=True, execution_dir_level='one')
    preprocessing_steps, evaluation_params_dict = construct_default_processing_evaluation_steps()

    # test_start_to_end_pipeline(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict, stop_after_alignment=True)
    test_start_to_end_pipeline(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict)
    # test_resume_analysis(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict)