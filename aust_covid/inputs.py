from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from general_utils.tex_utils import StandardTexDoc

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


def load_calibration_targets(
    start_request: datetime, 
    window: int,
    tex_doc: StandardTexDoc,
) -> tuple:
    
    description = 'Official COVID-19 data for Australian through 2022 were obtained from ' \
        '\href{https://www.health.gov.au/health-alerts/covid-19/weekly-reporting}{The Department of Health} ' \
        'on the 2\\textsuperscript{nd} of May 2023. Data that extended back to 2021 were obtained from ' \
        '\href{https://github.com/owid/covid-19-data/tree/master/public/data#license}{Our World in Data (OWID)} on ' \
        'the 16\\textsuperscript{th} of June 2023, downloaded from ' \
        'The final calibration target for cases was constructed as the OWID data for 2021 ' \
        'concatenated with the Australian Government data for 2022. ' \
        f'These daily case data were then smoothed using a {window}-day moving average. '
    tex_doc.add_line(description)

    # Australian national data
    national_data = pd.read_csv(DATA_PATH / 'Aus_covid_data.csv', index_col='date')
    national_data.index = pd.to_datetime(national_data.index)
    national_data = national_data[national_data['region'] == 'AUS']

    # OWID data
    owid_data = pd.read_csv(DATA_PATH / 'aust_2021_surv_data.csv', index_col=0)['new_cases']
    owid_data.index = pd.to_datetime(owid_data.index)

    # Join, truncate and smooth
    national_data_start = datetime(2022, 1, 1)
    interval = (start_request < owid_data.index) & (owid_data.index < national_data_start)
    composite_aust_data = pd.concat([owid_data[interval], national_data['cases']])
    return composite_aust_data.rolling(window=window).mean().dropna()


def load_who_data(
    window: int,
    tex_doc: StandardTexDoc,
) -> tuple:

    description = 'The daily time series of deaths for Australia was obtained from the ' \
        "World Heath Organization's \href{https://covid19.who.int/WHO-COVID-19-global-data.csv}" \
        '{Coronavirus (COVID-19) Dashboard} downloaded on 18\\textsuperscript{th} July 2023. ' \
        f'These daily deaths data were then smoothed using a {window}-day ' \
        'moving average. '
    tex_doc.add_line(description)

    raw_data = pd.read_csv(DATA_PATH / 'WHO-COVID-19-global-data.csv', index_col=0)
    processed_data = raw_data[raw_data['Country'] == 'Australia']
    processed_data.index = pd.to_datetime(processed_data.index)
    change_to_weekly_report_date = datetime(2022, 9, 16)
    processed_data = processed_data.loc[:change_to_weekly_report_date, :]
    death_data = processed_data['New_deaths']
    death_data = death_data.rolling(window=window).mean().dropna()

    return death_data, description


def load_serosurvey_data(immunity_lag):
    data = pd.Series(
        {
            datetime(2022, 2, 26): 0.207,
            datetime(2022, 6, 13): 0.554,
            datetime(2022, 8, 27): 0.782,
            datetime(2022, 12, 5): 0.850,
        }
    )
    data.index = data.index - timedelta(days=immunity_lag)

    description = 'We obtained estimates of the seroprevalence of antibodies to ' \
        'nucleocapsid antigen from Australia blood donors from Kirby Institute serosurveillance reports. ' \
        'Data are available from round 4 survey, available at https://www.kirby.unsw.edu.au/sites/default/files/documents/COVID19-Blood-Donor-Report-Round4-Nov-Dec-2022_supplementary%5B1%5D.pdf. ' \
        'Information on assay sensitivity is available at: https://www.kirby.unsw.edu.au/sites/default/files/documents/COVID19-Blood-Donor-Report-Round1-Feb-Mar-2022%5B1%5D.pdf' \
        f'We lagged these estimates by {immunity_lag} to account for the delay between infection and seroconversion. '

    return data, description


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


def load_google_mob_year_df(year=int) -> pd.DataFrame:
    mob_df = pd.read_csv(DATA_PATH / f"{year}_AU_Region_Mobility_Report.csv", index_col=8)
    mob_df = mob_df[[isinstance(region, float) for region in mob_df["sub_region_1"]]]  # National data subregion is given as nan
    mob_cols = [col for col in mob_df.columns if "percent_change_from_baseline" in col]
    mob_df = mob_df[mob_cols]
    mob_df.index = pd.to_datetime(mob_df.index)
    return mob_df
