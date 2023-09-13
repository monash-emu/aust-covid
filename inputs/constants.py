from datetime import datetime
from pathlib import Path
from plotly.express.colors import colorbrewer

REFERENCE_DATE = datetime(2019, 12, 31)
ANALYSIS_START_DATE = datetime(2021, 7, 1)
ANALYSIS_END_DATE = datetime(2022, 10, 1)
TARGETS_START_DATE = datetime(2021, 12, 15)
PLOT_START_DATE = datetime(2021, 12, 1)
WA_REOPEN_DATE = datetime(2022, 3, 3)
WHO_CHANGE_WEEKLY_REPORT_DATE = datetime(2022, 9, 16)

TARGETS_AVERAGE_WINDOW = 7
MOBILITY_AVERAGE_WINDOW = 7 
IMMUNITY_LAG = 14.0
N_LATENT_COMPARTMENTS = 4

MATRIX_LOCATIONS = [
    'school', 
    'home', 
    'work', 
    'other_locations',
]

MOBILITY_MAP = {
    'other_locations': 
        {
            'retail_and_recreation': 0.34, 
            'grocery_and_pharmacy': 0.33,
            'parks': 0.0,
            'transit_stations': 0.33,
            'workplaces': 0.0,
            'residential': 0.0,
        },
    'work':
        {
            'retail_and_recreation': 0.0, 
            'grocery_and_pharmacy': 0.0,
            'parks': 0.0,
            'transit_stations': 0.0,
            'workplaces': 1.0,
            'residential': 0.0,
        },  
}

PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / 'data'
SUPPLEMENT_PATH = PROJECT_PATH / 'supplement'
INPUTS_PATH = PROJECT_PATH / 'inputs'

AGE_STRATA = list(range(0, 80, 5))
STRAIN_STRATA = ['ba1', 'ba2', 'ba5']
INFECTION_PROCESSES = ['infection', 'early_reinfection', 'late_reinfection']

COLOURS = colorbrewer.Accent
CHANGE_STR = '_percent_change_from_baseline'
