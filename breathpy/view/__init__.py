try:
    from ..model.BreathCore import PeakAlignmentMethod, PeakDetectionMethod, ExternalPeakDetectionMethod, MccImsAnalysis, AnalysisResult, PerformanceMeasure
except ModuleNotFoundError:
    import os, sys

    current_working_dir = os.getcwd()
    project_prefix = ""
    for ele in sys.path:
        if ele.endswith(current_working_dir.split("/")[-1]):
            project_prefix = ele
    dir_to_add = project_prefix + "/breath/external/breathpy/breathpy"
    if dir_to_add not in sys.path:
        sys.path.append(dir_to_add)
        print("Adding {0} to pythonpath".format(dir_to_add))
    from ..model.BreathCore import PeakAlignmentMethod, PeakDetectionMethod, ExternalPeakDetectionMethod, MccImsAnalysis, AnalysisResult, PerformanceMeasure