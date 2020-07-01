import zipfile
from pathlib import Path

import joblib
import argparse

import tempfile
import os
from shutil import rmtree
import math
import numpy as np
import pandas as pd
import seaborn as sns

from .BreathCore import (MccImsMeasurement,
                              MccImsAnalysis,
                              PeakDetectionMethod,
                              ExternalPeakDetectionMethod,
                              NormalizationMethod,
                              PeakAlignmentMethod,
                              DenoisingMethod,
                              PerformanceMeasure,
                              FeatureReductionMethod,
                              AnalysisResult,
                              PredictionModel
                              )

from urllib.request import urlretrieve
from ..tools.tools import get_peax_binary_path

import matplotlib.pyplot as plt

def test_class_label_parsing(tmp_dir):
    labels1 = MccImsAnalysis.parse_class_labels(str(Path(tmp_dir)/"data/small_candy_anon/class_labels.txt"))
    labels2 = MccImsAnalysis.parse_class_labels(str(Path(tmp_dir)/"data/small_candy_anon/class_labels.tsv"))
    labels3 = MccImsAnalysis.parse_class_labels(str(Path(tmp_dir)/"data/small_candy_anon/class_labels.csv"))

    labels4 = MccImsAnalysis.parse_class_labels(str(Path(tmp_dir)/"data/small_candy_anon.zip/class_labels.csv"))
    # labels5 = MccImsAnalysis.parse_class_labels("/home/philipp/dev/breathpy/data/merged_candy/class_labels.zip/class_labels.tsv")
    labels_solution_str = {'BD18_1408280826_ims.csv': 'menthol', 'BD18_1408280834_ims.csv': 'citrus', 'BD18_1408280838_ims.csv': 'citrus', 'BD18_1408280841_ims.csv': 'menthol', 'BD18_1408280844_ims.csv': 'menthol', 'BD18_1408280851_ims.csv': 'citrus'}
    labels_tiny_solution_str = {'BD18_1408280826_ims.csv': 'menthol', 'BD18_1408280834_ims.csv': 'citrus'}
    assert labels1 == labels_tiny_solution_str
    assert labels2 == labels_tiny_solution_str
    assert labels3 == labels_solution_str
    assert labels4 == labels_solution_str
    # assert labels5 == labels_solution_str
    print("Passed Test for class_label parsing")



def test_percentage_reduction():
    # np.random.seed(1)
    # only works until a percentage of 50% - below the example construction inverse row property
    #   has a higher percentage and therefore will be present after filtering
    def prepare_feature_matrix(no_samples_per_class, no_features, lis_of_percentages, name_lis_of_features):
        assert (no_samples_per_class % 10) == 0
        assert not any([perc % 10 for perc in lis_of_percentages])
        assert no_features == len(lis_of_percentages)
        assert len(lis_of_percentages) == len(name_lis_of_features)

        sample_names = ["sample_{:03d}".format(i) for i in range(2*no_samples_per_class)]
        pos_sample_names = sample_names[:no_samples_per_class]
        neg_sample_names = sample_names[no_samples_per_class:]

        def prepare_percentage(percentage, mode=0):
            if mode:
                positives = [1]* int(no_samples_per_class * percentage / 100)
                negatives = [0]* int(math.ceil(no_samples_per_class * (1 - percentage / 100)))
            else:
                positives = [0] * int(no_samples_per_class * percentage / 100)
                negatives = [1] * int(math.ceil(no_samples_per_class * (1 - percentage / 100)))
            return positives + negatives

        def prep_one_class(class_sample_names, mod):
            lis_for_df = [prepare_percentage(perc, mod) for perc in lis_of_percentages]
            df = pd.DataFrame(lis_for_df).T
            df.index=class_sample_names
            df.columns = name_lis_of_features
            return df

        half_dfs = [prep_one_class(sn, mod) for sn, mod in zip([pos_sample_names, neg_sample_names], [0,1])]
        full_df = pd.concat(half_dfs)
        full_df['redundant_0'] = [0]*2*no_samples_per_class
        full_df['redundant_1'] = [1]*2*no_samples_per_class
        full_df['class'] = [0]*no_samples_per_class + [1]*no_samples_per_class
        #
        # print(full_df)
        # for class_i, group in full_df.groupby('class'):
        #     print("Class {}:\n".format(class_i), group.sum())
        return full_df


    matrix = prepare_feature_matrix(10, 3, [100,90,50], [100,90,50])

    class_labels = matrix['class'].values.tolist()
    trainings_matrix = matrix.drop('class', axis=1)

    column_mask_100 = MccImsAnalysis.remove_redundant_features_helper(trainings_matrix, class_labels, 1.0)
    column_mask_90 = MccImsAnalysis.remove_redundant_features_helper(trainings_matrix, class_labels,  .90)
    column_mask_60 = MccImsAnalysis.remove_redundant_features_helper(trainings_matrix, class_labels,  .60)
    column_mask_50 = MccImsAnalysis.remove_redundant_features_helper(trainings_matrix, class_labels,  .50)

    def prepare_expected_result(cols, bools):
        return pd.Series(index=cols, data=bools, dtype=bool)

    cols = trainings_matrix.columns.values
    val_100 = [1,0,0,0,1]
    val_90 = [1,1,0,0,1]
    val_60 = [1,1,0,0,1]
    val_50 = [1,1,1,0,1]

    assert np.all(column_mask_100 == prepare_expected_result(cols, val_100))
    assert np.all(column_mask_90 == prepare_expected_result(cols, val_90))
    assert np.all(column_mask_60 == prepare_expected_result(cols, val_60))
    assert np.all(column_mask_50 == prepare_expected_result(cols, val_50))
    print("Passed Test for percentage reduction")


def download_and_extract_test_set():
    tmp_dir = tempfile.gettempdir()
    # download example zip-archive
    zip0 = 'https://github.com/philmaweb/BreathAnalysis.github.io/raw/master/data/small_candy_anon.zip'
    zip_dst = Path(tmp_dir)/"data/small_candy_anon.zip"
    dst_dir = Path(tmp_dir)/"data/small_candy_anon/"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # also download other formats for class label parsing testing
    tsv = "https://github.com/philmaweb/BreathAnalysis.github.io/raw/master/class_labels.tsv"
    txt = "https://github.com/philmaweb/BreathAnalysis.github.io/raw/master/class_labels.txt"

    urlretrieve(zip0, zip_dst)
    urlretrieve(tsv, dst_dir/"class_labels.tsv")
    urlretrieve(txt, dst_dir/"class_labels.txt")

    # unzip archive into data subdirectory
    with zipfile.ZipFile(zip_dst, "r") as archive_handle:
        archive_handle.extractall(Path(dst_dir))
    return tmp_dir


def load_sample_normalized_measurement(tmp_dir):
    m = MccImsMeasurement(
        raw_filename=f"{tmp_dir}/data/small_candy_anon/BD18_1408280826_ims.csv")
    m.normalize_by_intensity()
    return m

def load_sample_raw_measurement(tmp_dir):
    return MccImsMeasurement(
        raw_filename=f"{tmp_dir}/data/small_candy_anon/BD18_1408280826_ims.csv")

def test_measurement_parsing(tmp_dir):
    # TODO check for attributes
    raw = load_sample_raw_measurement(tmp_dir)
    normed = load_sample_normalized_measurement(tmp_dir)

def run_peax():
    import subprocess
    m = load_sample_raw_measurement()
    peax_binary = get_peax_binary_path()
    raw_path = m.raw_filename
    result_path = m.raw_filename + "peax_out"
    subprocess.check_call([peax_binary, raw_path, result_path])

def main():
    tmpdir = download_and_extract_test_set()
    test_percentage_reduction()
    test_class_label_parsing(tmpdir)
    test_measurement_parsing(tmpdir)

if __name__ == '__main__':
    main()
