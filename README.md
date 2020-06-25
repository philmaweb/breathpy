# BreathPy

## Process breath samples of Multi-Capillary-Column Ion-Mobility-Spectrometry files
### Now with experimental support for GC/MS + LC/MS data

## Usage MCC/IMS

First prepare the example dataset by creating a subdirectory `data` and then linking the example files there.
```python
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# download example zip-archive
url = 'https://github.com/philmaweb/BreathAnalysis.github.io/raw/master/data/small_candy_anon.zip'
zip_dst = Path("data/small_candy_anon.zip")
dst_dir = Path("data/small_candy_anon/")
dst_dir.mkdir(parents=True, exist_ok=True)
urlretrieve(url, zip_dst)

# unzip archive into data subdirectory
with ZipFile(zip_dst, "r") as archive_handle:
    archive_handle.extractall(Path(dst_dir))
```   

Then run the example analysis like so:
```python
# import required functions
from breathpy.model.BreathCore import construct_default_parameters, construct_default_processing_evaluation_steps
from breathpy.model.CoreTest import test_start_to_end_pipeline

# define file prefix and default parameters
file_prefix = 'small_candy_anon'; folder_name = "small_candy_anon"

# assuming the data directory is in the current directory
plot_parameters, file_parameters = construct_default_parameters(file_prefix, folder_name, make_plots=True)

# create default parameters for preprocessing and evaluation
preprocessing_steps, evaluation_params_dict = construct_default_processing_evaluation_steps()

# call start
test_start_to_end_pipeline(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict)
```

For more complete examples see `model/CoreTest.py` `test_start_to_end_pipeline` and `test_resume_analysis`.
The `small_candy_anon` example is included in the package under `data/small_candy_anon/`, more coming soon.

## Usage GC/MS
Download and extract the example datasets into the current data subdirectory:
```bash
wget "https://github.com/bioinformatics-ca/bioinformatics-ca.github.io/raw/master/data_sets/Example_Jul0914_mzXML.zip"
wget "https://github.com/bioinformatics-ca/bioinformatics-ca.github.io/raw/master/data_sets/Example_Jul1114_mzXML.zip"
mkdir -p "data/eoe"
unzip Example_Jul1114_mzXML.zip -d data/eoe/
# overwrite the blank and alkstdt
unzip -o Example_Jul0914_mzXML.zip -d data/eoe/
# download class_labels.csv file
wget -O data/eoe/eoe_class_labels.csv "https://github.com/philmaweb/BreathAnalysis.github.io/raw/master/data/eoe_class_labels.csv"
```

```python
from pathlib import Path
import os
from breathpy.model.BreathCore import construct_default_parameters,construct_default_processing_evaluation_steps
from breathpy.model.ProcessingMethods import GCMSPeakDetectionMethod, PerformanceMeasure
from breathpy.model.GCMSTest import run_gcms_platform_multicore
from breathpy.generate_sample_data import generate_train_test_set_helper

"""
Runs analysis of Eosinophilic Esophagitis (EoE) sample set with 40 samples - gcms measurements
Dataset from https://bioinformaticsdotca.github.io/metabolomics_2018_mod2lab
:param cross_val_num:
:return:
"""
cross_val_num=5
# or use your local path to a dataset here
source_dir = Path(os.getcwd())/"data/eoe"
target_dir = Path(os.getcwd())/"data/eoe_out"

# will delete previous split and rewrite data
train_df, test_df = generate_train_test_set_helper(source_dir, target_dir, cross_val_num=5)
train_dir = Path(target_dir)/"train_eoe"

# prepare analysis
set_name = "train_eoe"
make_plots = True

# generate parameters
# if executing from breathpy directory use execution_dir_level='project',
plot_parameters, file_parameters = construct_default_parameters(set_name, set_name, make_plots=make_plots,
                                                                execution_dir_level='project')
preprocessing_params_dict = {GCMSPeakDetectionMethod.ISOTOPEWAVELET: {"hr_data": True}}
_, evaluation_params_dict = construct_default_processing_evaluation_steps(cross_val_num)

run_gcms_platform_multicore(sample_dir=train_dir, preprocessing_params=preprocessing_params_dict, evaluation_parms=evaluation_params_dict)
```
Also see `model/GCMSTest.py` for reference. 

### License
`BreathPy` is licensed under GPLv3, but contains binaries for PEAX, which is a free software for academic use only.
See
> [A modular computational framework for automated peak extraction from ion mobility spectra, 2014, D’Addario *et. al*](https://doi.org/10.1186/1471-2105-15-25)

## Contact
If you run into difficulties using `BreathPy`, please open an issue at our [GitHub](https://github.com/philmaweb/BreathPy) repository. Alternatively you can write an email to [Philipp Weber](mailto:pweber@imada.sdu.dk?subject=[BreathPy]%20BreathPy).