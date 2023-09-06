from pathlib import Path
import pandas as pd
import numpy as np
from plotly.express.colors import colorbrewer
from typing import Dict, Tuple

from summer2.parameters import Function
from summer2.utils import Epoch
from summer2.functions.time import get_linear_interpolation_function as linear_interp

from aust_covid.inputs import load_raw_pop_data, get_raw_state_mobility
from aust_covid.plotting import plot_state_mobility, plot_processed_mobility
from emutools.tex import StandardTexDoc
PROJECT_PATH = Path().resolve().parent
BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / 'supplement'
DATA_PATH = PROJECT_PATH / 'data'
CHANGE_STR = '_percent_change_from_baseline'
COLOURS = colorbrewer.Accent


def get_non_wa_mob_averages(
    state_data: pd.DataFrame, 
    mob_locs: set, 
    jurisdictions: set,
    tex_doc: StandardTexDoc,
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
    description = 'Values for Western Australia were extracted separately from the pooled data, ' \
        'while the data for the remaining states were linked to the same population size ' \
        'data as used to set the compartment sizes for the model. ' \
        'These population values were then used as weights to calculate weighted national averages ' \
        "for population mobility by each Google `location' " \
        f'(being {", ".join([i.replace("_percent_change_from_baseline", "").replace("_", " ") for i in mob_locs])}). '
    tex_doc.add_line(description, section='Mobility', subsection='Data processing')
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
    tex_doc: StandardTexDoc,
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
    description = 'Next, we used a mapping dictionary to map from the reported ' \
        "`locations' to the contact locations of the model's mixing matrix. "
    tex_doc.add_line(description, section='Mobility', subsection='Data processing')

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


def get_processed_mobility_data(
    tex_doc: StandardTexDoc
) -> pd.DataFrame:
    state_data = get_raw_state_mobility(tex_doc)
    jurisdictions, mob_locs = get_constants_from_mobility(state_data)
    
    fig = plot_state_mobility(state_data, jurisdictions, mob_locs, tex_doc)
    filename = 'state_mobility.jpg'
    fig.write_image(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        'Raw state mobility.', 
        filename,
        'Mobility',
    )

    wa_data = state_data.loc[state_data['sub_region_1'] == 'Western Australia', mob_locs]
    state_averages = get_non_wa_mob_averages(state_data, mob_locs, jurisdictions, tex_doc)

    description = 'Values were then converted from the reported percentage ' \
        'change from baseline to the proportional change relative to baseline. '
    tex_doc.add_line(description, section='Mobility', subsection='Data processing')
    non_wa_relmob = get_relative_mobility(state_averages)
    wa_relmob = get_relative_mobility(wa_data)

    mob_map = {
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

    for location in mob_map:
        if sum(mob_map[location].values()) != 1.0:
            raise ValueError(f'Mobility mapping does not sum to one for {location}')

    mob_map_table = pd.DataFrame(mob_map)
    mob_map_table.index = mob_map_table.index.str.replace('_', ' ')
    mob_map_table.columns = mob_map_table.columns.str.replace('_', ' ')
    tex_doc.include_table(mob_map_table, section='Mobility', subsection='Data processing')

    model_locs_mob = map_mobility_locations(wa_relmob, non_wa_relmob, mob_map, tex_doc)

    average_window = 7
    description = f'Next, we took the {average_window} moving average to smooth the ' \
        'often abrupt shifts in mobility, including with weekend and public holidays. '
    tex_doc.add_line(description, section='Mobility', subsection='Data processing')
    smoothed_mob = model_locs_mob.rolling(average_window).mean().dropna()

    description = 'Last, we squared the relative variations in mobility to account for ' \
        'the effect of reductions in visits to specific locations for both the infector ' \
        'and the infectee of the modelled social contacts. '
    tex_doc.add_line(description, section='Mobility', subsection='Data processing')
    model_mob = smoothed_mob ** 2.0

    fig = plot_processed_mobility(model_mob, smoothed_mob, tex_doc)
    filename = 'processed_mobility.jpg'
    fig.write_image(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        'Processed model mobility functions.', 
        filename,
        'Mobility',
    )

    return model_mob


def get_interp_funcs_from_mobility(
    mob_values: pd.DataFrame, 
    epoch: Epoch,
) -> Dict[str, Dict[str, Function]]:
    """
    Get summer-ready interpolated functions from mobility data.

    Args:
        mob_values: Mobility values by patch and contact location
        epoch: Model's epoch

    Returns:
        Nested dictionary of the interpolated functions
    """
    interp_funcs = {state: {} for state in mob_values.columns.get_level_values(0)}
    for state, mob_loc in mob_values.columns:
        interp_funcs[state][mob_loc] = linear_interp(epoch.dti_to_index(mob_values.index), mob_values[state, mob_loc].to_numpy())
    return interp_funcs


def get_dynamic_matrix(
    matrices: Dict[str, np.array], 
    mob_funcs: Function, 
    wa_prop_func: Function,
) -> np.array:
    """
    Construct dynamic matrix from scaling values by patch.

    Args:
        matrices: Unadjusted mixing matrices
        mob_funcs: Scaling functions by patch and contact location
        wa_prop_func: Scaling function for re-opening of WA

    Returns:
        Function to represent the scaling of the matrices with time-varying mobility
    """
    working_matrix = matrices['home'] + matrices['school']
    for location in ['other_locations', 'work']:
        for patch in ['wa', 'non_wa']:
            prop = wa_prop_func if patch == 'wa' else 1.0 - wa_prop_func
            working_matrix += matrices[location] * mob_funcs[patch][location] * prop
    return working_matrix
