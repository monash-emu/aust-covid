from pathlib import Path
import pandas as pd
import numpy as np
from plotly.express.colors import colorbrewer
from typing import Dict, Tuple

from aust_covid.inputs import load_raw_pop_data
PROJECT_PATH = Path().resolve().parent
DATA_PATH = PROJECT_PATH / 'data'
CHANGE_STR = '_percent_change_from_baseline'
COLOURS = colorbrewer.Accent


def get_non_wa_mob_averages(
    state_data: pd.DataFrame, 
    mob_locs: set, 
    jurisdictions: set,
) -> pd.DataFrame:
    """
    Calculate the weighted averages for the mobility estimates in the 
    states other than Western Australia.

    Args:
        state_data: Google data for all states
        mob_locs: Google mobility locations
        jurisdictions: Australian jurisdictions (states and territories)

    Returns:
        Weighted averages for non-WA jurisdictions
    """
    non_wa_data = state_data.loc[state_data['sub_region_1'] != 'Western Australia']

    # Add state population totals to dataframe
    state_pop_totals = load_raw_pop_data('31010do002_202206.xlsx').sum()
    for juris in jurisdictions:
        non_wa_data.loc[non_wa_data['sub_region_1'] == juris, 'weights'] = state_pop_totals[juris]

    # Calculate weighted averages
    state_averages = pd.DataFrame(columns=mob_locs)
    for mob_loc in mob_locs:
        state_averages[mob_loc] = non_wa_data.groupby(non_wa_data.index).apply(
            lambda x: np.average(x[mob_loc], weights=x['weights']),
        )
    return state_averages


def get_constants_from_mobility(
    state_data: pd.DataFrame,
) -> Tuple[set, set]:
    """
    Get the names of the jurisdictions (states and territories),
    and the Google mobility locations.

    Args:
        Google mobility data

    Returns:
        Names of jurisdictions and locations
    """
    jurisdictions = set([j for j in state_data['sub_region_1'] if j != 'Australia'])
    mob_locs = [c for c in state_data.columns if CHANGE_STR in c]
    return jurisdictions, mob_locs


def get_relative_mobility(
    mobility_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Google mobility values are provided as percentage change relative to baseline.
    However, we need mobility value relative to baseline estimate.

    Args:
        mobility_df: Data with values in original units

    Returns:
        Converted data
    """
    mobility_df.columns = [c.replace(CHANGE_STR, '') for c in mobility_df.columns]
    return 1.0 + mobility_df * 1e-2


def map_mobility_locations(
    wa_relmob: pd.DataFrame, 
    non_wa_relmob: pd.DataFrame, 
    mob_map: Dict[str, dict],
) -> pd.DataFrame:
    """
    Map mobility estimated values to model locations.

    Args:
        wa_relmob: Western Australia relative mobility
        non_wa_relmob: Rest of Australia relative mobility
        mob_map: User instructions for mapping

    Returns:
        Mobility functions for use in the model
    """
    patch_data = {
        'wa': wa_relmob,
        'non_wa': non_wa_relmob,
    }
    model_mob = pd.DataFrame(columns=pd.MultiIndex.from_product([patch_data.keys(), mob_map.keys()]))
    for patch in patch_data.keys():
        for mob_loc in mob_map.keys():
            data = patch_data[patch].assign(**mob_map[mob_loc]).mul(patch_data[patch]).sum(1)
            model_mob.loc[:, (patch, mob_loc)] = data
    return model_mob
