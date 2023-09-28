import pandas as pd
import numpy as np
from copy import copy
from datetime import datetime, timedelta
from plotly import graph_objects as go

from emutools.tex import get_tex_formatted_date, TexDoc, StandardTexDoc
from inputs.constants import TARGETS_START_DATE, TARGETS_AVERAGE_WINDOW, IMMUNITY_LAG, WHO_CHANGE_WEEKLY_REPORT_DATE, AGE_STRATA
from inputs.constants import DATA_PATH, SUPPLEMENT_PATH, NATIONAL_DATA_START_DATE

CHANGE_STR = '_percent_change_from_baseline'


def load_calibration_targets(tex_doc: StandardTexDoc) -> pd.Series:
    """
    See 'description' object text.

    Args:
        tex_doc: Documentation object

    Returns:
        Case targets
    """
    description = 'Official COVID-19 data for Australian through 2022 were obtained from ' \
        '\href{https://www.health.gov.au/health-alerts/covid-19/weekly-reporting}{The Department of Health} ' \
        f'on the {get_tex_formatted_date(datetime(2023, 5, 2))}. '
    tex_doc.add_line(description, 'Targets', subsection='Notifications')

    national_data = pd.read_csv(DATA_PATH / 'Aus_covid_data.csv', index_col='date')
    national_data.index = pd.to_datetime(national_data.index)
    national_data = national_data[national_data['region'] == 'AUS']
    return national_data['cases']


def load_owid_data(tex_doc: TexDoc) -> pd.Series:
    description = 'Data that extended back to 2021 were obtained from ' \
        '\href{https://github.com/owid/covid-19-data/tree/master/public/data#license}{Our World in Data (OWID)} on ' \
        f'the {get_tex_formatted_date(datetime(2023, 6, 16))}. '
    tex_doc.add_line(description, 'Targets', subsection='Notifications')

    owid_data = pd.read_csv(DATA_PATH / 'aust_2021_surv_data.csv', index_col=0)['new_cases']
    owid_data.index = pd.to_datetime(owid_data.index)
    return owid_data


def load_calibration_targets(tex_doc: TexDoc) -> tuple:
    description = 'The final calibration target for cases was constructed as the OWID data for 2021 ' \
        'concatenated with the Australian Government data for 2022. '
    tex_doc.add_line(description, 'Targets', subsection='Notifications')

def load_who_data(tex_doc: StandardTexDoc) -> pd.Series:
    """
    See 'description' object text.

    Args:
        tex_doc: Documentation object

    Returns:
        Death targets
    """
    description = 'The daily time series of deaths for Australia was obtained from the ' \
        "World Heath Organization's \href{https://covid19.who.int/WHO-COVID-19-global-data.csv}" \
        f'{{Coronavirus (COVID-19) Dashboard}} downloaded on {get_tex_formatted_date(datetime(2023, 7, 18))}. ' \
        f'These daily deaths data were then smoothed using a {TARGETS_AVERAGE_WINDOW}-day moving average. '
    tex_doc.add_line(description, 'Targets', subsection='Deaths')

    raw_data = pd.read_csv(DATA_PATH / 'WHO-COVID-19-global-data.csv', index_col=0)
    processed_data = raw_data[raw_data['Country'] == 'Australia']
    processed_data.index = pd.to_datetime(processed_data.index)
    processed_data = processed_data.loc[:WHO_CHANGE_WEEKLY_REPORT_DATE, :]
    death_data = processed_data['New_deaths']
    death_data = death_data.rolling(window=TARGETS_AVERAGE_WINDOW).mean().dropna()
    return death_data


def load_serosurvey_data(tex_doc: StandardTexDoc) -> pd.Series:
    """
    See 'description' object text.

    Args:
        tex_doc: Documentation object

    Returns:
        Serosurvey targets
    """
    description = 'We obtained estimates of the seroprevalence of antibodies to ' \
        'nucleocapsid antigen from Australia blood donors from Kirby Institute serosurveillance reports. ' \
        'Data are available from \href{https://www.kirby.unsw.edu.au/sites/default/files/documents/COVID19-Blood-Donor-Report-Round4-Nov-Dec-2022_supplementary%5B1%5D.pdf}' \
        '{the round 4 serosurvey}, with ' \
        '\href{https://www.kirby.unsw.edu.au/sites/default/files/documents/COVID19-Blood-Donor-Report-Round1-Feb-Mar-2022%5B1%5D.pdf}' \
        '{information on assay sensitivity also available}.' \
        f'We lagged these empiric estimates by {IMMUNITY_LAG} days to account for the delay between infection and seroconversion. '
    tex_doc.add_line(description, 'Targets', subsection='Seroprevalence')

    data = pd.Series(
        {
            datetime(2022, 2, 26): 0.207,
            datetime(2022, 6, 13): 0.554,
            datetime(2022, 8, 27): 0.782,
            datetime(2022, 12, 5): 0.850,
        }
    )
    data.index = data.index - timedelta(days=IMMUNITY_LAG)
    return data


def load_raw_pop_data(sheet_name: str) -> pd.DataFrame:
    """
    Load Australian population data from original spreadsheet.

    Args:
        sheet_name: Spreadsheet filenam

    Returns:
        Population data
    """
    skip_rows = list(range(0, 4)) + list(range(5, 227)) + list(range(328, 332))
    for group in range(16):
        skip_rows += list(range(228 + group * 6, 233 + group * 6))
    raw_data = pd.read_excel(DATA_PATH / sheet_name, sheet_name='Table_7', skiprows=skip_rows, index_col=[0])
    return raw_data


def load_pop_data(tex_doc: StandardTexDoc) -> pd.DataFrame:
    """
    See 'description' object text.

    Args:
        tex_doc: Documentation object

    Returns:
        Population by age and jurisdiction
    """
    sheet_name = '31010do002_202206.xlsx'
    sheet = sheet_name.replace('_', '\_')
    description = f'For estimates of the Australian population, the spreadsheet was downloaded ' \
        f'from the Australian Bureau of Statistics website on {get_tex_formatted_date(datetime(2023, 3, 1))} \cite{{abs2022}} ' \
        f"(sheet {sheet}). Minor jurisdictions other than Australia's eight major state and territories " \
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
    model_pop_data.index = AGE_STRATA
    return model_pop_data


def load_uk_pop_data(tex_doc: StandardTexDoc) -> pd.Series:
    """
    Get the UK census data. Data are in raw form,
    except for renaming the sheet to omit a space (from "Sheet 1"),
    to reduce the number of warnings.

    Args:
        tex_doc: Documentation object

    Returns:
        The population data
    """
    description = 'To align with the methodology of the POLYMOD study \cite{mossong2008} ' \
        'we sourced the 2001 UK census population for those living in the UK at the time of the census ' \
        'from the \href{https://ec.europa.eu/eurostat}{Eurostat database}. '
    tex_doc.add_line(description, 'Mixing')
    
    sheet_name = 'cens_01nscbirth__custom_6028079_page_spreadsheet.xlsx'
    data = pd.read_excel(
        DATA_PATH / sheet_name, 
        sheet_name='Sheet_1', 
        skiprows=list(range(0, 11)) + list(range(30, 37)), 
        usecols='B:C', 
        index_col=0,
    )
    data.index.name = 'age_group'
    data.columns = ['uk_pops']
    data.index = data.index.map(lambda string: string.replace('From ', '').replace(' years', ''))
    return data['uk_pops']


def load_household_impacts_data():
    filename = DATA_PATH / 'Australian Households, cold-flu-COVID-19 symptoms, tests, and positive cases in the past four weeks, by time of reporting .csv'
    data = pd.read_csv(filename, skiprows=[0] + list(range(5, 12)), index_col=0)
    data.columns = [col.replace(" (%)", "") for col in data.columns]
    index_map = {
        'A household member has symptoms of cold, flu or COVID-19 (a)': 'Proportion symptomatic',
        'A household member has had a COVID-19 test (b)': 'Proportion testing',
        'A household member who tested for COVID-19 was positive (c)(d)': 'Prop diagnosed with COVID-19',
    }
    data = data.rename(index=index_map)
    data = data.transpose()
    data.index = pd.to_datetime(data.index, format='%b-%y')
    return data


def get_ifrs(
    tex_doc: StandardTexDoc,
    show_figs=False,
) -> dict:
    description = 'Age-specific infection fatality rates (IFRs) were estimated by various groups ' \
        "in unvaccinated populations, including O'Driscoll and colleagues who estimated " \
        'IFRs using data from 45 countries. These IFRs pertained to the risk of death given infection ' \
        'for the wild-type strain of SARS-CoV-2 in unvaccinated populations, and so are unlikely to represent ' \
        'IFRs that would be applicable to the Australian population in 2022 because of vaccine-induced immunity ' \
        'and differences in severity for the variants we simulated. ' \
        'We therefore considered more recent studies, such as that of Erikstrup and colleagues to be better ' \
        'applicable to our local context, although also with limitations. ' \
        'Danish investigators used the increase in anti-nucleocapsid IgG seroprevalence in blood donors ' \
        'from January to April 2022 to estimate age-specific attack rates for the first Omicron wave in Denmark. ' \
        '\cite{erikstrup2022} They then re-weighted these values to estimate the attack rate ' \
        'for the general population aged 17-72. Linking this estimate to COVID-19 deaths ' \
        'reported within 60 days of a positive PCR, they estimated the Omicron-specific IFR, ' \
        'which was then re-weighted to exclude people with comorbidities. ' \
        'Therefore, their final results used in our analysis represent an Omicron-specific IFR ' \
        'for a healthy vaccinated population aged 17 to 72 years.' \
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

    # Erikstrup raw data
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


def get_raw_state_mobility(tex_doc: StandardTexDoc) -> pd.DataFrame:
    """
    Get raw Google mobility data, concatenating 2021 and 2022 data,
    retaining only state-level data and converting to date index.

    Returns:
    
    Returns:
        State-level mobility data, names of jurisdictions and locations
    """
    description = 'We undertook an alternative analysis in which estimates of population mobility ' \
        'were used to scale transmission rates. ' \
        'Raw estimates of Australian population mobility were obtained from Google, ' \
        'with 2021 and 2022 data concatenated together. '
    tex_doc.add_line(description, section='Mobility', subsection='Data processing')

    raw_data_2021 = pd.read_csv(DATA_PATH / '2021_AU_Region_Mobility_Report.csv', index_col=8)
    raw_data_2022 = pd.read_csv(DATA_PATH / '2022_AU_Region_Mobility_Report.csv', index_col=8)
    raw_data =  pd.concat([raw_data_2021, raw_data_2022])
    
    state_data = raw_data.loc[raw_data['sub_region_1'].notnull() & raw_data['sub_region_2'].isnull()]
    state_data.index = pd.to_datetime(state_data.index)

    jurisdictions = set([j for j in state_data['sub_region_1'] if j != 'Australia'])
    mob_locs = [c for c in state_data.columns if CHANGE_STR in c]
    return state_data, jurisdictions, mob_locs


def get_base_vacc_data() -> pd.DataFrame:
    """
    Get raw vaccination data obtained from Commonwealth.

    Returns:
        Collated vaccination data in its rawest form
    """
    vacc_df = pd.read_csv(DATA_PATH / 'aus_vax_data.csv', index_col=424)
    vacc_df.index = pd.to_datetime(vacc_df.index, infer_datetime_format=True)
    return vacc_df.sort_index()
