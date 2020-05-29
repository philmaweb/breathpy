from breathpy.model.BreathCore import construct_default_parameters, construct_default_processing_evaluation_steps
from breathpy.model.CoreTest import test_start_to_end_pipeline

# define file prefix and default parameters
file_prefix = 'small_candy_anon'; folder_name = "small_candy_anon"

# if executing from BreathAnalysis/model use execution_dir_level='one',
plot_parameters, file_parameters = construct_default_parameters(file_prefix, folder_name, make_plots=True, execution_dir_level='project')

# create default parameters for preprocessing and evaluation
preprocessing_steps, evaluation_params_dict = construct_default_processing_evaluation_steps()

# call start
test_start_to_end_pipeline(plot_parameters, file_parameters, preprocessing_steps, evaluation_params_dict)
