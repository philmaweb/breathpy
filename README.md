[![DOI](https://zenodo.org/badge/267952107.svg)](https://zenodo.org/badge/latestdoi/267952107)

# BreathPy
## A python library for breath gas biomarker profiling

## Installation

`BreathPy` depends on `python >=3.6`, but does **not yet** support `python==3.9`, as several dependencies are not yet available for python 3.9. It is available through `pip`. Make sure to activate your local virtual environment or use anaconda. To render decision trees we depend on the `graphviz` executable. Either install into your current environment using `pip install breathpy` or create, activate a new anaconda environment "breath" and install `breathpy` and `graphviz`:  
```bash
conda create --name breath python=3.8 pip graphviz -y
conda activate breath
pip install breathpy
```

If you want to use the tutorial jupyter notebooks - you also need to install jupyter `conda install jupyter`.

## Usage MCC-IMS

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
from breathpy.model.CoreTest import run_start_to_end_pipeline

# define file prefix and default parameters
file_prefix = folder_name = 'small_candy_anon'

# assuming the data directory is in the current directory
plot_parameters, file_parameters = construct_default_parameters(file_prefix, folder_name, make_plots=True)

# create default parameters for preprocessing and evaluation
preprocessing_steps, evaluation_params_dict = construct_default_processing_evaluation_steps()

# call start
run_start_to_end_pipeline(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict)
```

For more complete examples see `https://github.com/philmaweb/breathpy/blob/master/breathpy/tutorial/binary_candy.ipynb`, `https://github.com/philmaweb/breathpy/blob/master/breathpy/tutorial/multiclass_mouthwash.ipynb' or 'CoreTest.run_start_to_end_pipeline` and `CoreTest.run_resume_analysis`.
Example data is available at https://github.com/philmaweb/BreathAnalysis.github.io/tree/master/data.

## Usage GC-MS
### Now with experimental support for GC/MS + LC/MS data through pyOpenMS
Download and extract the example datasets into the current data subdirectory:
```python
# handle imports
from urllib.request import urlretrieve
from pathlib import Path
from zipfile import ZipFile

# download and extract data into data/algae directory
url = 'https://github.com/philmaweb/BreathAnalysis.github.io/raw/master/data/algae.zip'
zip_dst = Path("data/algae.zip")
dst_dir = Path("data/algae/")
dst_dir.mkdir(parents=True, exist_ok=True)
urlretrieve(url, zip_dst)

# unzip archive into data subdirectory
with ZipFile(zip_dst, "r") as archive_handle:
    archive_handle.extractall(Path(dst_dir))
```

```python
import os
from pathlib import Path
from breathpy.model.BreathCore import construct_default_parameters,construct_default_processing_evaluation_steps
from breathpy.model.ProcessingMethods import GCMSPeakDetectionMethod, PerformanceMeasure
from breathpy.model.GCMSTest import run_gcms_platform_multicore
from breathpy.generate_sample_data import generate_train_test_set_helper

"""
Runs analysis of the algae sample set (Sun M, Yang Z and Wawrik B (2018) Metabolomic Fingerprints 
of Individual Algal Cells Using the Single-Probe Mass Spectrometry Technique. 
Front. Plant Sci. 9:571. doi: 10.3389/fpls.2018.00571)

19 samples from four conditions - light, dark, nitrogen-limited and replete (post nitrogen-limited)
Samples originated from single-probe mass spectrometry files - we import created featureXML files.

:param cross_val_num:
:return:
"""
cross_val_num=3
# or use your local path to a dataset here
source_dir = Path("data/algae")
target_dir = Path("data")

# will delete previous split and rewrite data
train_df, test_df = generate_train_test_set_helper(source_dir, target_dir, cross_val_num=cross_val_num)
train_dir = Path(target_dir)/"train_algae"

# prepare analysis
set_name = "train_algae"
make_plots = True

# generate parameters
plot_parameters, file_parameters = construct_default_parameters(set_name, set_name, make_plots=make_plots)
preprocessing_params_dict = {GCMSPeakDetectionMethod.ISOTOPEWAVELET: {"hr_data": True}}
_, evaluation_params_dict = construct_default_processing_evaluation_steps(cross_val_num)

# running the full analysis takes less than 30 minutes of computation time using 6 cores - in this example most if not all computations are single core though
run_gcms_platform_multicore(
		sample_dir=train_dir, 
		preprocessing_params=preprocessing_params_dict, 
		evaluation_parms=evaluation_params_dict, num_cores=6)
```
Also see `model/GCMSTest.py` for reference. 

### License
`BreathPy` is licensed under GPLv3, but contains binaries for PEAX, which is a free software for academic use only.
See
> [A modular computational framework for automated peak extraction from ion mobility spectra, 2014, D’Addario *et. al*](https://doi.org/10.1186/1471-2105-15-25)

## Contact
If you run into difficulties using `BreathPy`, please open an issue at our [GitHub](https://github.com/philmaweb/BreathPy) repository. Alternatively you can write an email to [Philipp Weber](mailto:philmaweb+breathpy@gmail.com?subject=[BreathPy]%20BreathPy).
