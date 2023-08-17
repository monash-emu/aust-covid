from pathlib import Path
import pandas as pd
import numpy as np
from copy import copy
from datetime import datetime, timedelta
from general_utils.tex_utils import StandardTexDoc
from plotly import graph_objects as go

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / 'data'
SUPPLEMENT_PATH = BASE_PATH / 'supplement'


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
    tex_doc.add_line(description, 'Targets')

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
    tex_doc.add_line(description, 'Targets')

    raw_data = pd.read_csv(DATA_PATH / 'WHO-COVID-19-global-data.csv', index_col=0)
    processed_data = raw_data[raw_data['Country'] == 'Australia']
    processed_data.index = pd.to_datetime(processed_data.index)
    change_to_weekly_report_date = datetime(2022, 9, 16)
    processed_data = processed_data.loc[:change_to_weekly_report_date, :]
    death_data = processed_data['New_deaths']
    death_data = death_data.rolling(window=window).mean().dropna()

    return death_data


def load_serosurvey_data(
    immunity_lag: float,
    tex_doc: StandardTexDoc,
) -> pd.Series:

    description = 'We obtained estimates of the seroprevalence of antibodies to ' \
        'nucleocapsid antigen from Australia blood donors from Kirby Institute serosurveillance reports. ' \
        'Data are available from \href{https://www.kirby.unsw.edu.au/sites/default/files/documents/COVID19-Blood-Donor-Report-Round4-Nov-Dec-2022_supplementary%5B1%5D.pdf}' \
        '{the round 4 serosurvey}, with ' \
        '\href{https://www.kirby.unsw.edu.au/sites/default/files/documents/COVID19-Blood-Donor-Report-Round1-Feb-Mar-2022%5B1%5D.pdf}' \
        '{information on assay sensitivity also available}.' \
        f'We lagged these empiric estimates by {immunity_lag} days to account for the delay between infection and seroconversion. '
    tex_doc.add_line(description, 'Targets', subsection='Seroprevalence')

    data = pd.Series(
        {
            datetime(2022, 2, 26): 0.207,
            datetime(2022, 6, 13): 0.554,
            datetime(2022, 8, 27): 0.782,
            datetime(2022, 12, 5): 0.850,
        }
    )
    data.index = data.index - timedelta(days=immunity_lag)

    return data


def load_raw_pop_data(
    sheet_name: str,
):
    skip_rows = list(range(0, 4)) + list(range(5, 227)) + list(range(328, 332))
    for group in range(16):
        skip_rows += list(range(228 + group * 6, 233 + group * 6))
    raw_data = pd.read_excel(DATA_PATH / sheet_name, sheet_name='Table_7', skiprows=skip_rows, index_col=[0])
    return raw_data


def load_pop_data(
    age_strata: list,
    tex_doc: StandardTexDoc,
) -> tuple:
    sheet_name = '31010do002_202206.xlsx'
    description = f'For estimates of the Australian population, the spreadsheet was downloaded ' \
        'from the Australian Bureau of Statistics website on 01 March 2023.\cite{abs2022} ' \
        "Minor jurisdictions other than Australia's eight major state and territories " \
        '(i.e. Christmas island, the Cocos Islands, Norfolk Island and Jervis Bay Territory) are excluded from these data. ' \
        'These much smaller jurisdictions likely contribute little to overall COVID-19 epidemiology ' \
        'and are also unlikely to mix homogeneously with the larger states/territories. '
    tex_doc.add_line(description, 'Population')

    raw_data = load_raw_pop_data(sheet_name)
    spatial_pops = pd.DataFrame(
        {
            'wa': raw_data['Western Australia'], 
            'other': raw_data[[col for col in raw_data.columns if col not in ['Western Australia', 'Australia']]].sum(axis=1),
        }
    )
    model_pop_data = pd.concat([spatial_pops.loc[:'70-74'], pd.DataFrame([spatial_pops.loc['75-79':].sum()])])
    model_pop_data.index = age_strata

    return model_pop_data


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
    data.index = data.index.map(lambda string: string.replace('From ', '').replace(' years', ''))
    return data["uk_pops"]


def load_household_impacts_data():
    data = pd.read_csv(
        DATA_PATH / "Australian Households, cold-flu-COVID-19 symptoms, tests, and positive cases in the past four weeks, by time of reporting .csv",
        skiprows=[0] + list(range(5, 12)),
        index_col=0,
    )
    data.columns = [col.replace(" (%)", "") for col in data.columns]
    index_map = {
        "A household member has symptoms of cold, flu or COVID-19 (a)": "Proportion symptomatic",
        "A household member has had a COVID-19 test (b)": "Proportion testing",
        "A household member who tested for COVID-19 was positive (c)(d)": "Prop diagnosed with COVID-19",
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


def get_ifrs(tex_doc, show_figs=False):
    description = 'Age-specific infection fatality rates (IFRs) were estimated by various groups ' \
        "in unvaccinated populations, including O'Driscoll and colleagues who estimated " \
        'IFRs using data from 45 countries. These IFRs pertained to the risk of death given infection ' \
        'for the wild-type strain of SARS-CoV-2 in unvaccinated populations, and so are unlikely to represent ' \
        'IFRs that would be applicable to the Australian population in 2022 because of vaccine-induced immunity ' \
        'and differences in severity for the variants we simulated. ' \
        'We therefore considered more recent studies, such as that of Erikstrup and colleagues to be better ' \
        'applicable to our local context, although also with limitations. ' \
        '*** Insert brief description of Erikstrup study here. ***' \
        "As expected, the estimates from Erikstrup are consideraly lower than those of O'Driscoll. " \
        'However, there are also several potential differences between the Danish epidemic and that of Australia, ' \
        'most notably that community transmission had been established from much earlier in the pandemic in ' \
        'Denmark than in Australia. Further, given that these estimates estimate attack rates from blood donors, ' \
        'the age ranges covered by this study extend from 17 years to 73 years of age, making it necessary ' \
        'to extrapolate from these estimates to the extremes of age. ' \
        'We approached this extrapolation by identifying broadly equivalent younger and older age groups ' \
        'from each study for use as baselines for the more extreme age groups. Specifically, ' \
        'we considered that the IFR estimate for the 17 to 36 years-old age group from Erikstrup could ' \
        "be compared to the 25 to 29 years-old age group form O'Driscoll, and that " \
        'the 61 to 73 years-old age group from Erikstrup could be compared to the 65 to 69 years-old ' \
        "age group from O'Driscoll. We next calculated the ratio in the IFRs of these `equivalent' " \
        'age groups from each study, before then applying these ratios to the estimates from ' \
        "O'Driscoll for the age bands outside of the age range calculated by Erikstrup. " \
        '(i.e. 0-4, 5-9, 10-14, 15-19, 70-74, 75-79 and 80+). ' \
        'Next, to obtain IFR estimates for each modelled 5-year band from 75-79 years-old ' \
        'we performed linear interpolation from the estimates available to the mid-point of each modelled age band. ' \
        'We now have estimates for each 5-year band from 0-4 to 75-79 and for 80+ years-old. ' \
        'To calculate the IFR parameter for the modelled 75+ age band, we took an average of the 75-79 and 80+ ' \
        'estimates, weighted using the proportion of the Australian population aged 75+ who are aged ' \
        '75-79 and 80+. '
    tex_doc.add_line(description, 'Parameters', subsection='Infection Fatality Rates')
    
    # Raw data from O'Driscoll, 5-year age bands
    odriscoll = pd.Series(
        {
            0: 0.003,
            5: 0.001,
            10: 0.001,
            15: 0.003,
            20: 0.006,
            25: 0.013,
            30: 0.024,
            35: 0.04,
            40: 0.075,
            45: 0.121,
            50: 0.207,
            55: 0.323,
            60: 0.456,
            65: 1.075,
            70: 1.674,
            75: 3.203,
            80: 8.292,
        }
    ) / 100.0
    odriscoll.index = odriscoll.index + 2.5

    # Raw data from Erikstrup
    erikstrup = pd.Series(
        {
            (17 + 36) / 2: 2.6,
            (36 + 51) / 2: 5.8,
            (51 + 61) / 2: 14.6,
            (61 + 73) / 2: 24.6,
        }
    ) / 1e5

    # Most comparable upper and lower age group ratio
    lower_ratio = erikstrup[26.5] / odriscoll[27.5]
    upper_ratio = erikstrup[67.0] / odriscoll[67.5]

    # Apply the ratios to the upper and missing age groups without estimates from Erikstrup
    lower_adjusted = odriscoll[ :17.5] * lower_ratio
    upper_adjusted = odriscoll[72.5: ] * upper_ratio

    # Combine extrapolated estimates with Erikstrup
    combined = pd.concat([lower_adjusted, erikstrup, upper_adjusted])

    # Modelled breakpoints (lower values rather than midpoints of age groups)
    age_mid_points = np.linspace(2.5, 72.5, 15)
    final_values = pd.Series(np.interp(age_mid_points, combined.index, combined), index=age_mid_points)

    # Proportion of the 75+ age group IFR to take from the 80+ estimate
    pops = load_raw_pop_data('31010do002_202206.xlsx').sum(axis=1)
    prop_75_over_80 = pops['80-84': ].sum() / pops['75-79': ].sum()
    final_values[77.5] = combined[82.5] * prop_75_over_80 + combined[77.5] * (1.0 - prop_75_over_80)

    # Set age bands back to lower breakpoint values
    model_breakpoint_values = copy(final_values)
    model_breakpoint_values.index = final_values.index - 2.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=odriscoll.index, y=odriscoll, name="O'Driscoll"))
    fig.add_trace(go.Scatter(x=erikstrup.index, y=erikstrup, name='Erikstrup'))
    fig.add_trace(go.Scatter(x=upper_adjusted.index, y=upper_adjusted, name="Upper adjusted O'Driscoll"))
    fig.add_trace(go.Scatter(x=lower_adjusted.index, y=lower_adjusted, name="Lower adjusted O'Driscoll"))
    fig.add_trace(go.Scatter(x=combined.index, y=combined, name="Combined Erikstrup/O'Driscoll"))
    fig.add_trace(go.Scatter(x=final_values.index, y=final_values, name='Combined and interpolated'))
    fig.add_trace(go.Scatter(x=model_breakpoint_values.index, y=model_breakpoint_values, name='Values by model breakpoints'))
    fig.update_yaxes(type='log')
    ifr_fig_name = 'ifr_calculation.jpg'
    fig.write_image(SUPPLEMENT_PATH / ifr_fig_name)
    tex_doc.include_figure(
        'Illustration of the calculation of the base age-specific infection-fatality rates applied in the model. ',
        ifr_fig_name,
        'Parameters', 
        'Infection Fatality Rates',
    )

    if show_figs:
        fig.show()

    model_breakpoint_values.index = model_breakpoint_values.index.map(lambda i: f'ifr_{int(i)}')
    return model_breakpoint_values.to_dict()
