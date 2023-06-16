import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


def load_calibration_targets(
    start_request: datetime, 
    rolling_window: int=7,
) -> pd.Series:

    # Australian national data
    national_data = pd.read_csv(DATA_PATH / "Aus_covid_data.csv", index_col="date")
    national_data.index = pd.to_datetime(national_data.index)
    national_data = national_data[national_data["region"] == "AUS"]

    # OWID data
    owid_data = pd.read_csv(DATA_PATH / "aust_2021_surv_data.csv", index_col=0)["new_cases"]
    owid_data.index = pd.to_datetime(owid_data.index)

    # Join them together, truncate and smooth
    national_data_start = datetime(2022, 1, 1)
    window = (start_request < owid_data.index) & (owid_data.index < national_data_start)
    composite_aust_data = pd.concat([owid_data[window], national_data["cases"]])
    final_data = composite_aust_data.rolling(window=rolling_window).mean().dropna()

    description = "Official COVID-19 data for Australian through 2022 were obtained from https://www.health.gov.au/health-alerts/covid-19/weekly-reporting " \
        "on the 2nd of May 2023. " \
        "Data that extended back to 2021 were obtained from Our World in Data (attribution: https://github.com/owid/covid-19-data/tree/master/public/data#license)" \
        "on the 16th of June 2023, downloaded from " \
        "https://github.com/owid/covid-19-data/blob/master/public/data/jhu/full_data.csv and filtered to the Australian data only." \
        "The final calibration target for cases was constructed as the OWID data for 2021 concatenated with the Australian Government data for 2022. "

    return final_data, description


def load_pop_data() -> tuple:
    """
    Get the Australian population data from ABS source.

    Returns:
        The population data
        The name of the sheet
    """
    skip_rows = list(range(0, 4)) + list(range(5, 227)) + list(range(328, 332))
    for group in range(16):
        skip_rows += list(range(228 + group * 6, 233 + group * 6))
    sheet_name = "31010do002_202206.xlsx"
    data = pd.read_excel(DATA_PATH / sheet_name, sheet_name="Table_7", skiprows=skip_rows, index_col=[0])
    return data, sheet_name


def load_uk_pop_data() -> pd.Series:
    """
    Get the UK census data. Data are in raw form,
    except that renaming the sheet to omit a space (from "Sheet 1")
    results in many fewer warnings.

    Returns:
        The population data
    """
    sheet_name = "cens_01nscbirth__custom_6028079_page_spreadsheet.xlsx"
    data = pd.read_excel(
        DATA_PATH / sheet_name, 
        sheet_name="Sheet_1", 
        skiprows=list(range(0, 11)) + list(range(30, 37)), 
        usecols="B:C", 
        index_col=0,
    )
    data.index.name = "age_group"
    data.columns = ["uk_pops"]
    return data["uk_pops"]


def load_household_impacts_data():
    "https://www.abs.gov.au/statistics/people/people-and-communities/household-impacts-covid-19-survey/latest-release"
    data = pd.read_csv(
        DATA_PATH / "Australian Households, cold-flu-COVID-19 symptoms, tests, and positive cases in the past four weeks, by time of reporting .csv",
        skiprows=[0] + list(range(5, 12)),
        index_col=0,
    )
    data.columns = [col.replace(" (%)", "") for col in data.columns]
    index_map = {
        "A household member has symptoms of cold, flu or COVID-19 (a)": "Proportion symptomatic",
        "A household member has had a COVID-19 test (b)": "Proportion testing",
        "A household member who tested for COVID-19 was positive (c)(d)": "Proportion diagnosed with COVID-19",
    }
    data = data.rename(index=index_map)
    data = data.transpose()
    data.index = pd.to_datetime(data.index, format="%b-%y")
    return data


def load_param_info(
    data_path: Path, 
    param_names: dict,
) -> pd.DataFrame:
    """
    Load specific parameter information from 
    a ridigly formatted yaml file or crash otherwise.

    Args:
        data_path: Location of the source file
        param_names: The parameters provided

    Returns:
        The parameters info DataFrame contains the following fields:
            descriptions: A brief reader-digestible name/description for the parameter
            units: The unit of measurement for the quantity (empty string if dimensionless)
            evidence: TeX-formatted full description of the evidence underpinning the choice of value
    """
    data_cols = ["descriptions", "units", "evidence"]
    param_keys = param_names.keys()
    out_df = pd.DataFrame(index=param_keys, columns=data_cols)
    with open(data_path, "r") as param_file:
        all_data = yaml.safe_load(param_file)
        for col in data_cols:
            working_data = all_data[col]
            if param_keys != working_data.keys():
                raise ValueError("Incorrect keys for data")
            out_df[col] = working_data.values()
    return out_df
