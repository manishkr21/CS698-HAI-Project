"""
Configuration constants
"""
from pathlib import Path

# Paths
ARTIFACT_FOLDER = Path('artifacts')
ARTIFACT_FOLDER.mkdir(exist_ok=True)

# Constants
RANDOM_STATE = 42
TARGET_COL = 'Target'
SENSITIVE_FEATURE = 'Gender'
CLASS_NAMES = ['Graduate', 'Dropout', 'Enrolled'] 

UNBIASED_FEATURES = [
    "Daytime/evening attendance", "Admission grade", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date",
    "Scholarship holder", "Age at enrollment", "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)"
]

FEATURE_NAMES = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender",
    "Scholarship holder", "Age at enrollment", "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]

# Fairlearn availability check
try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.metrics import MetricFrame, demographic_parity_difference, demographic_parity_ratio
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("Warning: Fairlearn not installed. Bias mitigation features disabled.")
