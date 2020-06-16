import multiprocessing
import subprocess
import glob
from shutil import copyfile


def list_all_ims_files(raw_dir):
    """
    Lists all files ending with _ims.csv in raw_dir
    :return:
    """
    return glob.glob(raw_dir + "*_ims.csv")

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
        return (command_list, subprocess.check_call(command_list, stderr=subprocess.STDOUT), None)
    except Exception as e:
        return (command_list, None, e)

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

    p = multiprocessing.Pool(1)
    # save all output and error codes in a list
    # configuration file needs to be in the same directory as the execution level of the python script
    # - bad parsing from peax, we cannot pass arguments directly and it doesnt accept another path
    call_history = [(used_command, output, error) for used_command, output, error in
                    p.map(execute_command, command_list)]  # provide all commands to execute
    return call_history


def compare_peax_results(fn1, fn2, df1=None, df2=None):
    # read in
    # sort by retention time, then irm
    # compare irm val + intensity
    # peak id - don't care
    import pandas as pd

    if df1 is None:
        try:
            df1 = pd.read_csv(fn1, sep="\t", index_col=0)
            df1 = df1.sort_values(by=['retention_time', 'inverse_reduced_mobility'])[['retention_time', 'inverse_reduced_mobility', 'intensity']].reindex()
        except KeyError:  # we have peax raw result
            df1 = pd.read_csv(fn1, sep="\t", index_col=0)
            df1 = df1.rename(columns={"r":'retention_time', "t":'inverse_reduced_mobility', "signal":'intensity'})
    df1 = df1.sort_values(by=['retention_time', 'inverse_reduced_mobility'])[
                ['retention_time', 'inverse_reduced_mobility', 'intensity']].reindex()
    if df2 is None:
        df2 = pd.read_csv(fn2, sep="\t", index_col=0)
    df2 = df2.sort_values(by=['retention_time', 'inverse_reduced_mobility'])[['retention_time', 'inverse_reduced_mobility', 'intensity']].reindex()

    # now compare
    comp_lis= []
    for i in range(df1.shape[0]):
        row1 = df1.iloc[[i]]
        row2 = df2.iloc[[i]]
        row_comp = all(row1 == row2)
        if not row_comp:
            print(f"row {i}", row_comp)
        comp_lis.append(row_comp)
    print(f"df1 == df2, {all(comp_lis)}")

    # # compare if import function is problem
    # from breath.external.breathpy.tools.tools import compare_peax_results
    # from celery.contrib import rdb;
    # df2 = mcc_ims_analysis.peak_detection_results['PEAX'][0].peak_df
    # df2.index = df2['measurement_name']
    # fn1 = "/home/philipp/Downloads/test/newest/BD18_1408280834_ims.csv_PEAX_peak_detection_result.csv"
    # compare_peax_results(fn1=fn1, fn2="", df2=df2)
    # rdb.set_trace()


if __name__ == "__main__":

    peax_binary = "/home/philipp/dev/breathpy/bin/peax1.0-LinuxX64/peax"
    raw_dir = "/home/philipp/dev/breathpy/bin/peax1.0-LinuxX64/"
    raw_files = list_all_ims_files(raw_dir)

    # # find out what parameters used for best prediction results
    # for i in range(1,8):
    #     # overwrite config file
    #     cfg_fn = f"parameters.cfg.{i}"
    #     copyfile(cfg_fn, "parameters.cfg")
    #
    #     out_names = []
    #     for raw_file in raw_files:
    #         out_names.append(f"{raw_file}_out.csv.{i}")
    #     peax_files(path_to_peax_binary=peax_binary, in_filenames=raw_files, out_filenames=out_names)
    # copyfile("parameters.cfg.orig", "parameters.cfg")

    compare_peax_results("BD18_1408280838_ims.csv_PEAX_peak_detection_result.csv", 'BD18_1408280838_ims.csv_PEAX_peak_detection_result.csv.goal')
    compare_peax_results("BD18_1408280838_ims.csv_out.csv.7", 'BD18_1408280838_ims.csv_PEAX_peak_detection_result.csv.goal')