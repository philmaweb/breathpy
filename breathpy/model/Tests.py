import pdb
import glob
import zipfile
from pathlib import Path

import joblib

import argparse

import tempfile
import os
from shutil import rmtree
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors

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


from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

def test_class_label_parsing():
    labels1 = MccImsAnalysis.parse_class_labels("/home/philipp/dev/breathpy/data/merged_candy/class_labels.txt")
    labels2 = MccImsAnalysis.parse_class_labels("/home/philipp/dev/breathpy/data/merged_candy/class_labels.tsv")
    labels3 = MccImsAnalysis.parse_class_labels("/home/philipp/dev/breathpy/data/merged_candy/class_labels.csv")

    labels4 = MccImsAnalysis.parse_class_labels("/home/philipp/dev/breathpy/data/merged_candy/class_labels.zip/class_labels.csv")
    labels5 = MccImsAnalysis.parse_class_labels("/home/philipp/dev/breathpy/data/merged_candy/class_labels.zip/class_labels.tsv")
    labels_solution_str = {'BD18_1408280826_ims.csv': 'menthol', 'BD18_1408280834_ims.csv': 'citrus', 'BD18_1408280838_ims.csv': 'citrus', 'BD18_1408280841_ims.csv': 'menthol', 'BD18_1408280844_ims.csv': 'menthol', 'BD18_1408280851_ims.csv': 'citrus', 'BD18_1511121654_ims.csv': 'menthol', 'BD18_1511121658_ims.csv': 'citrus', 'BD18_1511121702_ims.csv': 'citrus', 'BD18_1511121706_ims.csv': 'menthol', 'BD18_1511121709_ims.csv': 'citrus', 'BD18_1511121712_ims.csv': 'citrus', 'BD18_1511121716_ims.csv': 'citrus', 'BD18_1511121719_ims.csv': 'menthol', 'BD18_1511121723_ims.csv': 'menthol', 'BD18_1511121727_ims.csv': 'citrus', 'BD18_1511121730_ims.csv': 'menthol', 'BD18_1511121734_ims.csv': 'menthol', 'BD18_1511121738_ims.csv': 'menthol', 'BD18_1511121742_ims.csv': 'citrus'}
    assert labels1 == labels_solution_str
    assert labels2 == labels_solution_str
    assert labels3 == labels_solution_str
    assert labels4 == labels_solution_str
    assert labels5 == labels_solution_str
    print("Passed Test for class_label parsing")



def test_percentage_reduction():
    # np.random.seed(1)
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
    val_100 = [1,0,0,0,0]
    val_90 = [1,1,0,0,0]
    val_60 = [1,1,0,0,0]
    val_50 = [1,1,1,0,0]

    assert np.all(column_mask_100 == prepare_expected_result(cols, val_100))
    assert np.all(column_mask_90 == prepare_expected_result(cols, val_90))
    assert np.all(column_mask_60 == prepare_expected_result(cols, val_60))
    assert np.all(column_mask_50 == prepare_expected_result(cols, val_50))
    print("Passed Test for percentage reduction")


def load_sample_normalized_measurement():
    m = MccImsMeasurement(raw_filename="/home/philipp/PycharmProjects/django_mockup/mysite/breath/external/breathpy/data/merged_candy/BD18_1408280826_ims.csv")
    m.normalize_by_intensity()
    return m

def load_sample_raw_measurement():
    return MccImsMeasurement(raw_filename="/home/philipp/PycharmProjects/django_mockup/mysite/breath/external/breathpy/data/merged_candy/BD18_1408280826_ims.csv")

def run_peax():
    import subprocess
    m = load_sample_raw_measurement()
    peax_binary = "/home/philipp/PycharmProjects/django_mockup/mysite/breath/external/breathpy/bin/peax1.0-LinuxX64/peax"
    raw_path = m.raw_filename
    result_path = m.raw_filename + "peax_out"
    subprocess.check_call([peax_binary, raw_path, result_path])

def main():
    test_percentage_reduction()
    test_class_label_parsing()

if __name__ == '__main__':
    main()
