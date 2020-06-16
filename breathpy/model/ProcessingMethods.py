from abc import ABCMeta
from enum import Enum, auto
### MCCIMS PREPROCESSING METHODS

class MethodSelection(Enum):
    """
    Base Method Selection
    """
    __metaclass__ = ABCMeta

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self):
        return self.name

    # if implementing __eq__ we also need to implement __hash__ otherwise it cannot be use in dicts andymore
    # https://stackoverflow.com/questions/1608842/types-that-define-eq-are-unhashable
    def __eq__(self, other):
        if isinstance(other, Enum):
            return self.value == other.value
        if isinstance(other, str):
            return str(self) == other
        else:
            return object.__eq__(Enum, other)

    def __hash__(self):
        return hash(self._name_)

class PreprocessingMethod(MethodSelection):
    """
    Base Preprocessing method
    """

class MccImsPreprocessingMethod(PreprocessingMethod):
    """
    Could be a PeakDetectionMethod, NormalizationMethod, DenoisingMethod or PeakAlignmentMethod
    """

class MeasurementAlignmentScalingMethod(MccImsPreprocessingMethod):
    """
    Scale measurements on retention time and inverse reduced mobility axis
    Used to reduce variation between instruments, laboratories and operation parameters
    Will be performed in Analysis
    """
    IRM_SCALING = auto()
    # additional method could be irm-alignment


class PeakDetectionMethod(MccImsPreprocessingMethod):
    """
    Available peak detection Methods for MCC Analysis.
    Will be performed in Analysis
    """
    WATERSHED = auto()
    TOPHAT = auto()
    JIBB = auto()
    VISUALNOWLAYER = auto()

    # look at possible methods at https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
    # eg scipy.signal.find_peaks_cwt
    # or https://gist.github.com/endolith/250860


class ExternalPeakDetectionMethod(MccImsPreprocessingMethod):
    """
    External peak detection Methods for MCC Analysis.
    Depend on third party libraries and external binaries - eg PEAX
    Will be performed in Analysis
    """
    PEAX = auto()
    CUSTOM = auto()
    # PEAX does it's own normalization, need to write Normalization parameters to parameters.cfg in directory from where peax is executed
    # DeepPeak


class DenoisingMethod(MccImsPreprocessingMethod):
    """
    Method to remove Noise from an MccImsMeasurement.
    Will be performed in Measurement
    """
    NOISE_SUBTRACTION = auto()
    MEDIAN_FILTER = auto()
    SAVITZKY_GOLAY_FILTER = auto()
    GAUSSIAN_FILTER = auto()
    CROP_INVERSE_REDUCED_MOBILITY = auto()
    DISCRETE_WAVELET_TRANSFORMATION = auto()



class NormalizationMethod(MccImsPreprocessingMethod):
    """
    Available normalization Methods for MCC Analysis
    Will be performed in Measurement
    """
    BASELINE_CORRECTION = auto()
    INTENSITY_NORMALIZATION = auto()


class PeakAlignmentMethod(MccImsPreprocessingMethod):
    """
    Available PeakAlignment Methods for MCC Analysis
    Mainly Clustering methods
    Will be performed in Analysis
    """
    WARD_CLUSTERING = auto()
    K_MEANS_CLUSTERING = auto()
    AFFINITY_PROPAGATION_CLUSTERING = auto()
    WINDOW_FRAME = auto()
    MEAN_SHIFT_CLUSTERING = auto()
    DB_SCAN_CLUSTERING = auto()
    PROBE_CLUSTERING = auto()
    # add Voronoi diagram of known substances

class PerformanceMeasure(MethodSelection):
    """
    Available PerformanceMeasure Methods for MCC-IMS / GCMS Analysis
    Will be performed in Analysis
    """
    RANDOM_FOREST_CLASSIFICATION = auto()  # with ROC and Decision tree of best features
    PCA_LDA_CLASSIFICATION = auto()  # with ROC and Decision tree of best features
    FDR_CORRECTED_P_VALUE = auto()  # Boxplots with p_values
    DECISION_TREE_TRAINING = auto()  # only affects decision tree, not classification of samples

class FeatureReductionMethod(MethodSelection):
    """
    FeatureReduction Methods
    """
    REMOVE_PERCENTAGE_FEATURES = auto()  # remove features that are not present in a certain percentage of measurements
    # REMOVE_CONTROL_PEAKS_POSITIVE = auto()  #  option to only keep features that are more intense in comp to control samples
    # REMOVE_CONTROL_PEAKS_NEGATIVE = auto()  #  option to only keep features that are less intense in comp to control samples
    # REMOVE_PRE_RIP_PEAKS = auto()  # option to only keep features that have irm >= 0.485


### GCMS PREPROCESSING METHODS

class GCMSPreprocessingMethod(PreprocessingMethod):
    """
    Preprocessing Method applicable for GCMS data - TODO include LCMS in class name?
    """

class GCMSPeakDetectionMethod(GCMSPreprocessingMethod):
    """
    Peak detection method for GCMS data - extract the features from profile or centroided data - also see https://abibuilder.informatik.uni-tuebingen.de/archive/openms/Documentation/nightly/html/a16047.html
    """
    CENTROIDED = auto() # FeatureFinderCentroided from pyopenms
    ISOTOPEWAVELET = auto() # FeatureFinderIsotopeWavelet from pyopenms -  "is only able to handle MS1 data"
    CUSTOM = auto()
    # TODO implement smoothing function, as isotopewavelet is slow and overly sensitive
    # TODO MRM? MULTIPLEX?
    # TODO FeatureFinderAlgorithmSH = SuperHirn
    #       centroids data, then does feature finding and feature merging
    #       extracts 1D peaks though

class GCMSAlignmentMethod(GCMSPreprocessingMethod):
    """
    Peak alignement method for GCMS data - align featureXML files into a consensus map
    """
    # TODO add different options
    POSE_FEATURE_GROUPER = auto()  # align features on axis with pose clustering, then use feature grouper to cluster - peak_ids are generated based on 2 decimals in rt, mz and


class GCMSSupportedDatatype(MethodSelection):
    """
    Supported datatypes Methods
    """
    RAW_MZML_OR_MZXML = auto()
    CENTROIDED_MZML_OR_MZXML = auto()
