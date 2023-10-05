from typing import Dict, List
from datetime import timedelta
import pandas as pd
from jax import numpy as jnp
import numpy as np
from datetime import datetime
import scipy
from scipy.optimize import minimize

from summer2.parameters import Function, Data, Time

from inputs.constants import VACC_AVERAGE_WINDOW, VACC_IMMUNE_DURATION, IMMUNITY_LAG


def calc_rates_for_interval(
        start_props: pd.core.series.Series, end_props: pd.core.series.Series, delta_t: float, strata: List[str],
        active_flows: dict
) -> dict:
    """
    Calculate the transition rates associated with each inter-stratum flow for a given time interval.

    The system can be described using a linear ordinary differential equations such as:
    X'(t) = M.X(t) ,
    where M is the transition matrix and X is a column vector representing the proportions over time

    The solution of this equation is X(t) = exp(M.t).X_0,
    where X_0 represents the proportions at the start of the time interval.

    The transition parameters informing M must then verify the following equation:
    X(t_end) = exp(M.delta_t).X_0,
    where t_end represents the end of the time interval.

    Args:
        start_props: user-requested stratum proportions at the start of the time interval
        end_props: user-requested stratum proportions at the end of the time interval
        delta_t: width of the time interval
        strata: list of strata
        active_flows: Dictionary listing the flows driving the inter-stratum transitions. Keys are flow names and values
        are length-two tuples representing the flows' sources and destinations.
    Returns:
        The estimated transition rates stored in a dictionary using the flow names as keys.

    """
    # Determine some basic characteristics
    n_strata = len(strata)
    n_params = len(active_flows)
    ordered_flow_names = list(active_flows.keys())

    # Create the function that we need to find the root of
    def function_to_zero(params):
        # params is a list ordered in the same order as ordered_flow_names

        # Create the transition matrix associated with a given set of transition parameters
        m = np.zeros((n_strata, n_strata))
        for i_row, stratum_row in enumerate(strata):
            for i_col, stratum_col in enumerate(strata):
                if i_row == i_col:
                    # Diagonal components capture flows starting from the associated stratum
                    relevant_flow_names = [f_name for f_name, f_ends in active_flows.items() if f_ends[0] == stratum_row]
                    for f_name in relevant_flow_names:
                        m[i_row, i_col] -= params[ordered_flow_names.index(f_name)]
                else:
                    # Off-diagonal components capture flows from stratum_col to stratum_row
                    for f_name, f_ends in active_flows.items():
                        if f_ends == (stratum_col, stratum_row):
                            m[i_row, i_col] = params[ordered_flow_names.index(f_name)]

        # Calculate the matrix exponential, accounting for the time interval width
        exp_mt = scipy.linalg.expm(m * delta_t)

        # Calculate the difference between the left and right terms of the equation
        diff = np.matmul(exp_mt, start_props) - end_props

        # Return the norm of the vector to make the minimised function a scalar function
        return scipy.linalg.norm(diff)

    # Define bounds to force the parameters to be positive
    bounds = [(0., None)] * n_params

    # Numerical solving
    solution = minimize(function_to_zero, x0=np.zeros(n_params), bounds=bounds, method="TNC")

    return {ordered_flow_names[i]: solution.x[i] for i in range(len(ordered_flow_names))}


def get_vacc_data_masks(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Get masks for vaccination dataframe to identify the various programs.

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
) -> pd.DataFrame:
    """
    Add the additional columns needed in the vaccination dataframe that are not included as raw data.

    Args:
        df: The vaccination dataframe returned by get_base_vacc_data
        rolling_window: Period over which to calculate the rolling average
        immune_duration: Duration of immunity over which to look back for vaccination

    Returns:
        Augmented vaccination dataframe with additional columns
    """
    masks = get_vacc_data_masks(df)
    df['adult booster'] = df.loc[:, masks['age 16+, 3+ doses'] + masks['age 16+, 4+ doses']].sum(axis=1)
    df['primary full'] = df[masks['age 5-11, 2+ doses']].sum(axis=1)
    df = df.drop(datetime(2022, 7, 8))
    df.loc[df['primary full'] < 0.0, 'primary full'] = 0.0
    df['adult booster smooth'] = df.loc[:, 'adult booster'].rolling(VACC_AVERAGE_WINDOW).mean()
    df['primary full smooth'] = df.loc[:, 'primary full'].rolling(VACC_AVERAGE_WINDOW).mean()
    df['incremental adult booster'] = df['adult booster smooth'].diff()
    df['incremental primary full'] = df['primary full smooth'].diff()
    df['prop incremental adult booster'] = df['incremental adult booster'] / df['National - Population 16 and over']
    df['prop incremental primary full'] = df['incremental primary full'] / df['National - Population 5-11']
    df['boosted in preceding'] = df['incremental adult booster'].rolling(VACC_IMMUNE_DURATION).sum()
    df['primary full in preceding'] = df['incremental primary full'].rolling(VACC_IMMUNE_DURATION).sum()
    df['prop boosted in preceding'] = df['boosted in preceding'] / df['National - Population 16 and over']
    df['prop primary full in preceding'] = df['primary full in preceding'] / df['National - Population 5-11']
    return df


def piecewise_constant(x, breakpoints, values):
    return values[sum(x >= breakpoints)]


def get_model_vacc_vals_from_data(
    vacc_df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Extract, tidy and adapt data needed from vaccination dataframe.

    Args:
        vacc_df: Vaccination data including calculated boosting proportions

    Returns:
        Just the series of the booster coverage needed by the model
    """
    data = vacc_df[column].dropna()
    data.index += timedelta(days=IMMUNITY_LAG)
    return data[~data.index.duplicated(keep='first')]


def get_rate_from_prop(
    prop1: float, 
    prop2: float, 
    duration: float,
) -> float:
    """
    Calculate the transition rate needed to achieve a requested end proportion
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


def calc_vacc_funcs_from_props(
    data: pd.Series, 
    epoch,
) -> Function:
    """
    Get transition function for moving population between immune and non-immune categories.

    Args:
        data: Vaccination coverage values over time index
        epoch: Model's epoch

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
