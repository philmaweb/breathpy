# BreathPy Project

## Process MCC/IMS data and experimental support for GC/MS + LC/MS data

## Usage  

### MCC/IMS

```python
from breathpy.model.BreathCore import construct_default_parameters, construct_default_processing_evaluation_steps
from breathpy.model.CoreTest import test_start_to_end_pipeline

# define file prefix and default parameters
file_prefix = 'small_candy_anon'; folder_name = "small_candy_anon"

# if executing from breathpy/model use execution_dir_level='one'
plot_parameters, file_parameters = construct_default_parameters(file_prefix, folder_name, make_plots=True, execution_dir_level='one')

# create default parameters for preprocessing and evaluation
preprocessing_steps, evaluation_params_dict = construct_default_processing_evaluation_steps()

# call start
test_start_to_end_pipeline(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict)
```

For more complete examples see `model/CoreTest.py` `test_start_to_end_pipeline` and `test_resume_analysis`.
Example data coming soon.

### GC/MS
```python
from pathlib import Path
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
# use your local path to a dataset here 
source_dir = "/home/philipp/Desktop/gcms_data/canada_2018/eoe/"
target_dir = "/home/philipp/Desktop/gcms_data/canada_2018/eoe_out/"

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
See `model/GCMSTest.py` for reference. Example data coming soon.

### License
BreathPy is licensed under GPLv3, but contains binaries for PEAX, which is a free software for academic use only.
See
> [A modular computational framework for automated peak extraction from ion mobility spectra, 2014, D’Addario *et. al*](https://doi.org/10.1186/1471-2105-15-25)

## Contact
If you run into difficulties using BreathPy, please open an issue at our [GitHub](https://github.com/philmaweb/BreathPy) repository. Alternatively you can write an email to [Philipp Weber](mailto:pweber@imada.sdu.dk?subject=[BreathPy-PyPI]%20BreathPy).