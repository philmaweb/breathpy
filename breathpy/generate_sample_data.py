import glob
import matplotlib
# important - would lead to a segmentation fault if importing pyplot otherwise
matplotlib.use("Agg")

from collections import Counter
from io import StringIO, BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copy as file_copy
from shutil import rmtree
from sklearn.model_selection import train_test_split
from zipfile import ZipFile, ZIP_DEFLATED

from .model.BreathCore import MccImsAnalysis, MccImsMeasurement


def generate_full_candy_classes(plot_params, file_params, preprocessing_steps, evaluation_params_dict):
    all_files = glob.glob(file_params['data_dir'] + "*_ims.csv")

    class_labels = {r[0]: r[1] for r in np.loadtxt(file_params['data_dir']+"class_labels.csv", delimiter=",", dtype=str, skiprows=1)}
    from collections import OrderedDict
    class_labels = OrderedDict(class_labels)
    analysis = MccImsAnalysis([MccImsMeasurement(f) for f in all_files],
                              preprocessing_steps, [],
                              performance_measure_parameters=evaluation_params_dict,
                              class_label_file=file_params['label_filename'], dataset_name='full_candy', dir_level="")

    # analysis.preprocess()

    for m in analysis.measurements:
        class_label = class_labels.get(m.filename)
        m.set_class_label(class_label)
        # HeatmapPlot.FastIntensityMatrix(m,plot_params,title=class_label)
    from sklearn.model_selection import train_test_split
    from shutil import copy as file_copy
    from shutil import rmtree
    from pathlib import Path

    X = [k for k in class_labels.keys()]
    y = [v for v in class_labels.values()]
    # class_labels[m.filename]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    train_df = pd.DataFrame({"name":X_train, "candy":y_train})
    test_df = pd.DataFrame({"name":X_test, "candy":y_test})
    train_dir = file_params['data_dir'] + "train_full_candy/"
    test_dir = file_params['data_dir'] + "test_full_candy/"

    # delete train and test dir
    if Path(train_dir).exists():
        rmtree(train_dir, ignore_errors=True)
    if Path(test_dir).exists():
        rmtree(test_dir, ignore_errors=True)

    # create directory
    Path(train_dir).mkdir()
    Path(test_dir).mkdir()

    # create class_label file
    train_df[['name', 'candy']].to_csv(train_dir+"class_labels.csv", sep=",")
    test_df[['name', 'candy']].to_csv(test_dir+"class_labels.csv", sep=",")

    # copy files to target dirs
    for fn in train_df.name:
        file_path = file_params['data_dir'] + fn
        new_path = train_dir + fn
        file_copy(file_path, dst=new_path)

    # same for test set
    for fn in test_df.name:
        file_path = file_params['data_dir'] + fn
        new_path = test_dir + fn
        file_copy(file_path, dst=new_path)


def generate_train_test_set_helper(sample_dir, target_dir, cross_val_num=5, seed=42, has_just_feature_matrix=False, dataset_name=""):
    sample_dir_path = Path(sample_dir)
    target_dir_path = Path(target_dir)

    if not dataset_name:
        dataset_name = sample_dir_path.stem

    # guess class label file
    class_label_file = MccImsAnalysis.guess_class_label_extension(sample_dir_path)
    class_labels = MccImsAnalysis.parse_class_labels(class_label_file)

    available_raw_files = sorted(sample_dir_path.glob("*_ims.csv"))
    available_preprocessed_files = sorted(sample_dir_path.glob("*_ims_preprocessed.csv"))
    available_pdrs = sorted(sample_dir_path.glob("*_peak_detection_result.csv"))

    # make sure we know which files we're missing so we can get them - too many files is not a problem - subset is allowed - or if we need to edit class labels
    not_available_raw_files = []
    if available_raw_files:
        raw_files = []
        for arw in available_raw_files:
            raw_files.append(arw.name)
        raw_file_set = set(raw_files)

        for fn in class_labels.keys():
            if fn not in raw_file_set:
                not_available_raw_files.append(fn)
        print(f"Missing raw files: {not_available_raw_files}")

    not_available_preproc_files = []
    if available_preprocessed_files:
        preproc_files = []
        for apf in available_preprocessed_files:
            raw_name = apf.name.split("_ims.csv_")[0]
            raw_name += "_ims.csv"
            preproc_files.append(raw_name)
        preproc_file_set = set(preproc_files)
        for fn in class_labels:
            if fn not in preproc_file_set:
                not_available_preproc_files.append(f"{fn[:-4]}_preprocessed.csv")
        print(f"Missing preprocessed files: {not_available_preproc_files}")

    # approximate search, dont want to spell out all pdr_names
    not_available_pdr_files = []
    if available_pdrs:
        av_pdrs = []
        for apdr in available_pdrs:
            raw_name = apdr.name.split("_ims.csv_")[0]
            raw_name += "_ims.csv"
            av_pdrs.append(raw_name)

        av_pdr_set = set(av_pdrs)
        for fn in class_labels:
            if fn not in av_pdr_set:
                not_available_pdr_files.append(f"{fn[:-4]}_peak_detection_result.csv")
        print(f"Missing peak detection result: {not_available_pdr_files}")

    if not_available_raw_files or not_available_preproc_files or not_available_pdr_files:
        raise ValueError("Class labels needs to be adjusted or missing files added.")

    # check if we have a layer_file
    potential_layers = [str(filename) for filename in sample_dir_path.glob("*") if
                        (str.endswith(str(filename), "layer.csv") or str.endswith(str(filename), "layer.xls"))]

    print(f"Preparing dataset for {Counter(class_labels.values())} using {cross_val_num}-fold cross validation splits.")

    X = [k for k in class_labels.keys()]
    y = [v for v in class_labels.values()]
    # class_labels[m.filename]

    test_fraction = 1. / cross_val_num
    train_df, test_df = split_labels_ratio(class_labels, train_val_fraction=1-test_fraction, seed=seed)

    train_dir = str(target_dir_path) + "/" + f"train_{dataset_name}/"
    test_dir = str(target_dir_path) + "/" + f"test_{dataset_name}/"

    print(f"Deleting {train_dir} and {test_dir}")
    # delete train and test dir if already exisitent
    if Path(train_dir).exists():
        rmtree(train_dir, ignore_errors=True)
    if Path(test_dir).exists():
        rmtree(test_dir, ignore_errors=True)

    # TODO also remove exsiting results, such as peak_detection_results, feature matrices
    print(f"Creating {train_dir} and {test_dir}")

    Path(train_dir).mkdir(parents=True)
    Path(test_dir).mkdir(parents=True)

    tr_class_label_fn = Path(train_dir)/"class_labels.csv"
    te_class_label_fn = Path(test_dir)/"class_labels.csv"

    train_df[['name', 'label']].to_csv(tr_class_label_fn, sep=",", index=False)
    test_df[['name', 'label']].to_csv(te_class_label_fn, sep=",", index=False)

    # check if it has peak detection results
    pdrs = sorted(sample_dir_path.glob("*_peak_detection_result.csv"))

    # distribute into train and test list
    train_name_set = set(train_df['name'].values)
    test_name_set = set(test_df['name'].values)

    cannot_copy = []
    for pdr in pdrs:
        raw_fn_pre = pdr.name.split("_ims.csv")[0]
        raw_fn = raw_fn_pre + "_ims.csv"

        new_fn = ""
        if raw_fn in train_name_set:
            new_fn = Path(train_dir)/pdr.name
        elif raw_fn in test_name_set:
            new_fn = Path(test_dir)/pdr.name
        else:
            cannot_copy.append(pdr)

        # copy to destination
        if new_fn:
            file_copy(pdr,new_fn)

    if cannot_copy:
        print(f"{len(cannot_copy)} PDRs not in either index.", f"{cannot_copy}")

    if has_just_feature_matrix:
        # write feature matrix
        potential_feature_matrices = sample_dir_path.glob("*_feature_matrix.csv")
        for fn in potential_feature_matrices:

            try:
                fm = pd.read_csv(fn, index_col=0)
                tr_fm = fm.loc[fm.index.intersection(train_df['name'])]
                te_fm = fm.loc[fm.index.intersection(test_df['name'])]

                tr_fm_fn = Path(train_dir)/"train_feature_matrix.csv"
                te_fm_fn = Path(test_dir)/"test_feature_matrix.csv"

                tr_fm.to_csv(tr_fm_fn)
                te_fm.to_csv(te_fm_fn)
                print(f"Created feature matrices {tr_fm_fn} and {te_fm_fn}")

                # also implement for other branches - pdr and preprocessed
                for t_dir, t_fm, t_cl, in zip([train_dir, test_dir], [tr_fm_fn, te_fm_fn],
                                              [tr_class_label_fn, te_class_label_fn]):
                    t_dir_path = Path(t_dir)
                    t_dir_name = t_dir_path.stem

                    zip_path_tr = t_dir_path / f"{t_dir_name}.zip"
                    with ZipFile(zip_path_tr, 'w', ZIP_DEFLATED) as trzip:
                        trzip.write(t_fm, t_fm.name)  # needs to exist as file object on disk to to write to zip
                        trzip.write(t_cl, t_cl.name)

            except ValueError:
                # go until no more potential candidates, which should be just one anyways
                pass

    else:
        # copy files to target dirs - only works if raw files are actually there, not always the case - eg if there's just results
        raw_files_not_copied = []
        for fn in train_df.name:
            file_path = Path(sample_dir_path)/fn
            new_path = Path(train_dir)/fn
            if file_path.exists():
                file_copy(file_path, dst=new_path)
            else:
                raw_files_not_copied.append(file_path)
        # same for test set
        for fn in test_df.name:
            file_path = Path(sample_dir_path)/fn
            new_path = Path(test_dir)/fn
            if file_path.exists():
                file_copy(file_path, dst=new_path)
            else:
                raw_files_not_copied.append(file_path)
        if raw_files_not_copied:
            print(f"Didn't copy {len(raw_files_not_copied)} raw files - as not found in source directory.")

    # guess layer file and copy to target dir too
    if potential_layers:
        potential_layer_file = potential_layers[0]
        layer_name = Path(potential_layer_file).stem + Path(potential_layer_file).suffix
        file_copy(potential_layers[0], dst=str(train_dir) + "/" + layer_name)
        file_copy(potential_layers[0], dst=str(test_dir) + "/" + layer_name)

    print(f"{'|' * 40}\nFinished preparation of {dataset_name}\n")
    return train_df, test_df


def split_labels_ratio(class_label_dict, train_val_fraction, seed=42):
    """
    Split class labels into train and validation using stratified split
    """
    X = [k for k in class_label_dict.keys()]
    y = [v for v in class_label_dict.values()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_val_fraction, random_state=seed,
                                                        stratify=y)  # , shuffle=True)  # will always create same split

    # labels might not be sorted
    X_tr_args = np.argsort(X_train)
    X_te_args = np.argsort(X_test)

    X_train = np.take_along_axis(np.array(X_train), X_tr_args, axis=0)
    X_test = np.take_along_axis(np.array(X_test), X_te_args, axis=0)

    y_train = np.take_along_axis(np.array(y_train), X_tr_args, axis=0)
    y_test = np.take_along_axis(np.array(y_test), X_te_args, axis=0)

    train_df = pd.DataFrame({"name": X_train, "label": y_train})
    test_df = pd.DataFrame({"name": X_test, "label": y_test})
    return train_df, test_df


def write_raw_files_to_zip(raw_filenames, class_label_dict, origin_zip_fn):
    """
    Write passed `raw_filenames`, `class_label_dict` from `origin_zip_fn` to `target_zip` archive
    returns: buffer
    """
    # make a csv file and write to buffer

    class_label_df = pd.DataFrame(list(class_label_dict.items()), columns=["name", "label"])
    class_label_buffer = StringIO()
    class_label_df[['name', 'label']].to_csv(class_label_buffer, sep=",", index=False)

    # write to target zip
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as target_zip:
        target_zip.writestr("class_labels.csv", class_label_buffer.getvalue())

        # write from origin_zip
        with ZipFile(origin_zip_fn, 'r') as origin_zip:
            # get buffer from zip - and rewrite to in memory zip
            for fn_to_extract in raw_filenames:
                with origin_zip.open(fn_to_extract) as zopen:
                    target_zip.writestr(fn_to_extract, zopen.read())

    return buffer

def generate_train_test_sets(dir_full_set, root_target_dir, cross_val_num=5, seed=42):

    return generate_train_test_set_helper(dir_full_set, root_target_dir, cross_val_num, seed)