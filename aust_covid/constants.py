from datetime import datetime
from pathlib import Path
from plotly.express.colors import colorbrewer

REFERENCE_DATE = datetime(2019, 12, 31)
ANALYSIS_START_DATE = datetime(2021, 7, 1)
ANALYSIS_END_DATE = datetime(2022, 10, 1)
TARGETS_START_DATE = datetime(2021, 12, 15)
PLOT_START_DATE = datetime(2021, 11, 30)
WA_REOPEN_DATE = datetime(2022, 3, 3)
WHO_CHANGE_WEEKLY_REPORT_DATE = datetime(2022, 9, 16)
NATIONAL_DATA_START_DATE = datetime(2022, 1, 1)

TARGETS_AVERAGE_WINDOW = 7
MOBILITY_AVERAGE_WINDOW = 7
IMMUNITY_LAG = 14.0
NUCLEOCAPS_SENS = 0.78
VACC_IMMUNE_DURATION = 100
N_LATENT_COMPARTMENTS = 4

MATRIX_LOCATIONS = [
    "school",
    "home",
    "work",
    "other_locations",
]

MOBILITY_MAP = {
    "other_locations": {
        "retail_and_recreation": 0.34,
        "grocery_and_pharmacy": 0.33,
        "parks": 0.0,
        "transit_stations": 0.33,
        "workplaces": 0.0,
        "residential": 0.0,
    },
    "work": {
        "retail_and_recreation": 0.0,
        "grocery_and_pharmacy": 0.0,
        "parks": 0.0,
        "transit_stations": 0.0,
        "workplaces": 1.0,
        "residential": 0.0,
    },
}

AUST_COVID_PATH = Path(__file__).parent.resolve()
DATA_PATH = AUST_COVID_PATH / "data"
INPUTS_PATH = AUST_COVID_PATH / "data"

AGE_STRATA = list(range(0, 80, 5))
STRAIN_STRATA = ["ba1", "ba2", "ba5"]
IMMUNITY_STRATA = ["imm", "nonimm"]
INFECTION_PROCESSES = ["infection", "early_reinfection", "late_reinfection"]

COLOURS = colorbrewer.Accent
CHANGE_STR = "_percent_change_from_baseline"

RUN_IDS = {
    "none": "2023-11-02T1101-none-d50k-t10k-b5k",
    "mob": "2023-11-01T1547-mob-d50k-t10k-b5k",
    "vacc": "2023-11-02T1102-vacc-d50k-t10k-b5k",
    "both": "2023-11-02T1103-both-d50k-t10k-b5k",
}
ANALYSIS_FEATURES = {
    "none": {
        "mob": False,
        "vacc": False,
    },
    "mob": {
        "mob": True,
        "vacc": False,
    },
    "vacc": {
        "mob": False,
        "vacc": True,
    },
    "both": {
        "mob": True,
        "vacc": True,
    },
}
PRIMARY_ANALYSIS = "mob"

BURN_IN = 25000
OPTI_DRAWS = 100

_PROJECT_PATH = None


def set_project_base_path(path: Path):
    global _PROJECT_PATH
    _PROJECT_PATH = Path(path).resolve()

    return get_project_paths()


def get_project_paths():
    if _PROJECT_PATH is None:
        raise Exception(
            "set_project_base_path must be called before attempting to use project paths"
        )
    return {
        "PROJECT_PATH": _PROJECT_PATH,
        "SUPPLEMENT_PATH": _PROJECT_PATH / "supplement",
        "RUNS_PATH": _PROJECT_PATH / "runs",
        "OUTPUTS_PATH": _PROJECT_PATH / "outputs",
        "DATA_PATH": _PROJECT_PATH / "aust_covid" / "data",
    }
