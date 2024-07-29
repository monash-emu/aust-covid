from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from copy import deepcopy
import pandas as pd
from jax import numpy as jnp
import numpy as np

from summer2.parameters import Function, Data, Time
from summer2.utils import Epoch

from aust_covid.constants import IMMUNITY_LAG


def get_vacc_data_masks(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Get masks for vaccination dataframe to identify the various programs.

    Args:
        df: The vaccination dataframe returned by get_base_vacc_data

    Returns:
        Dictionary containing lists of the names of columns relevant to each program
    """
    masks = {
        'age 16+, 2+ doses': [
            c for c in df.columns if 'Number of people fully vaccinated' in c and 
            not any(s in c for s in [' - M - ', ' - F - '])
        ],
        'age 16+, 3+ doses': [
            col for col in df.columns if 
            'National' in col and 
            any([s in col for s in ['who have received 3 doses', 'with 3 or more doses', 'with more than two doses', 'with more than 2 doses']]) and 
            any([s in col for s in ['16', '18']]) and
            not any([s in col for s in ['increase', 'Percentage', 'Indigenous']])
        ],
        'age 16+, 4+ doses': [
            col for col in df.columns if
            'National' in col and
            any([s in col for s in ['Winter Boosters number', 'who have received 4 doses', 'Fourth dose number']]) and
            not any([s in col for s in ['30', '65', 'increase']])
        ],
        'age 12-15, 2+ doses': [
            col for col in df.columns if
            '12-15' in col and
            any([s in col for s in ['National', 'Age group']]) and
            any([s in col for s in ['2 doses', 'fully vaccinated']]) and
            not any([s in col for s in ['Indigenous', 'Population', '- F -', '- M -']])
        ],
        'age 5-11, 2+ doses': [
            col for col in df.columns if
            'National' in col and
            '5-11' in col and
            any([s in col for s in ['2 doses', 'fully vaccinated']])
        ],
    }
    masks['age 16+, 2+ doses'].sort()
    return masks


def add_derived_data_to_vacc(
    df: pd.DataFrame, 
) -> Tuple[pd.DataFrame]:
    """Add the additional columns needed in the vaccination dataframe that are not included as raw data.
    Lastly, lag all fields for development of immunity and return both unlagged and lagged dataframes.

    Args:
        df: The vaccination dataframe returned by get_base_vacc_data
        rolling_window: Period over which to calculate the rolling average
        immune_duration: Duration of immunity over which to look back for vaccination

    Returns:
        Augmented vaccination dataframe with additional columns
        Additional fields added are:
            primary full: Numbers of 5-11 year-olds receiving second doses
            adult booster: Numbers of 16+ year-olds receiving third or subsequent doses
            prop primary full: 'primary full' as proportional coverage
            prop adult booster: 'adult booster' as proportional coverage
            prop remaining primary full: Proportion receiving primary dose over time interval
            prop remaining adult booster: Proportion receiving booster dose over time interval
            duration: Period since the previous time step
            rate primary full: Per capita primary dose transition rate to move people between model strata
            rate adult booster: Per capita adult booster transition rate
    """
    masks = get_vacc_data_masks(df)
    df['primary full'] = df[masks['age 5-11, 2+ doses']].sum(axis=1)
    df['adult booster'] = df.loc[:, masks['age 16+, 3+ doses'] + masks['age 16+, 4+ doses']].sum(axis=1)
    df = df.drop(datetime(2022, 7, 8))
    df = df[~df.index.duplicated(keep='first')]
    df['prop primary full'] = df['primary full'] / df['National - Population 5-11']
    df['prop adult booster'] = df['adult booster'] / df['National - Population 16 and over']
    df['prop remaining primary full'] = df['prop primary full'].diff() / (1.0 - df['prop primary full'])
    df['prop remaining adult booster'] = df['prop adult booster'].diff() / (1.0 - df['prop adult booster'])
    df['duration'] = [0] + [(df.index[i + 1] - df.index[i]).days for i in range(len(df) - 1)]
    df.loc[df['prop remaining primary full'] < 0.0, 'prop remaining primary full'] = 0.0
    df['rate primary full'] = df['prop remaining primary full'] / df['duration']
    df['rate adult booster'] = df['prop remaining adult booster'] / df['duration']

    lagged_df = deepcopy(df)
    lagged_df.index = lagged_df.index + timedelta(days=IMMUNITY_LAG)
    return df, lagged_df


def get_full_vacc_props(
    vacc_df: pd.DataFrame, 
    masks: List[str],
) -> pd.DataFrame:
    """Get coverage of full vaccination by age group from main vaccination data,
    retaining column names, even though these are now proportions.

    Args:
        vacc_df: The main vaccination data dataframe with original and derived fields
        masks: The names of the columns pertaining to second dose in adults

    Returns:
        Separate dataframe containing the coverage values
    """
    full_df = pd.DataFrame()
    for full_vacc_mask in masks:
        age_str = full_vacc_mask.replace('Age group - ', '').replace('- Number of people fully vaccinated', '').replace(' ', '')
        full_df[full_vacc_mask] = vacc_df[full_vacc_mask] / vacc_df[f'Age group - {age_str} - Population']
    return full_df


def get_rate_from_prop(
    prop1: float, 
    prop2: float, 
    duration: float,
) -> float:
    """
    Calculate the transition rate needed to achieve a requested end proportion,
    given a specific starting proportion and duration for two model strata.
    Equation solves the expression:
        (1 - prop2) / (1 - prop1) = exp(-rate * duration)

    Args:
        prop1: Starting proportion
        prop2: Ending proportion
        duration: Time interval between targeted proportions

    Returns:
        Per capita transition rate
    """
    return (np.log(1.0 - prop1) - np.log(1.0 - prop2)) / duration


def piecewise_constant(
    time: Time, 
    breakpoints: Data, 
    values: Data,
) -> float:
    """Get the required value from an array according to the time point
    at which to index it.

    Args:
        time: Time point of interest
        breakpoints: Series of breakpoints in time
        values: The values for each time interval

    Returns:
        The value corresponding to the time point of interest
    """
    return values[sum(time >= breakpoints)]


def calc_vacc_funcs_from_props(
    data: pd.Series, 
    epoch: Epoch,
) -> Function:
    """Get transition function for moving population between immune and non-immune categories.

    Args:
        data: Vaccination coverage values over time index
        epoch: Epidemiological model's epoch

    Returns:
        Piecewise constant function for implementation as flow between strata
    """

    # Get rates from data
    rates_df = []
    for i_date, date in enumerate(data.index[:-1]):
        next_date = data.index[i_date + 1]
        rates_df.append(get_rate_from_prop(data[date], data[next_date], (next_date - date).days))
        
    # Get functions from rates
    time_vals = Data(jnp.array([*epoch.datetime_to_number(data.index)]))
    vals = Data(jnp.array((0.0, *rates_df, 0.0)))
    return Function(piecewise_constant, [Time, time_vals, vals])        
