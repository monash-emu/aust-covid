import pandas as pd
import numpy as np
from typing import Dict

from summer2.parameters import Function
from summer2.utils import Epoch
from summer2.functions.time import get_linear_interpolation_function as linear_interp

from aust_covid.inputs import load_raw_pop_data, get_raw_state_mobility
from emutools.tex import StandardTexDoc
from .constants import MOBILITY_MAP, MOBILITY_AVERAGE_WINDOW, CHANGE_STR


def get_non_wa_mob_averages(
    state_data: pd.DataFrame,
    mob_locs: set,
    jurisdictions: set,
    tex_doc: StandardTexDoc,
    cross_ref: bool = True,
) -> pd.DataFrame:
    """Calculate the weighted averages for the mobility estimates in the
    states other than Western Australia.

    Args:
        state_data: Google data for all states
        mob_locs: Google mobility locations
        jurisdictions: Australian jurisdictions (states and territories)
        cross_ref: Whether to include cross references in the document

    Returns:
        Weighted averages for non-WA jurisdictions
    """
    fig_ref = " (Figure \\ref{input_population})" if cross_ref else ""
    description = (
        "Values for Western Australia were extracted separately from the pooled data, "
        "while the data for the remaining states were linked to the same population size "
        f"data as used to set the compartment sizes for the model{fig_ref}. "
        "These population values were then used as weights to calculate weighted national averages "
        "for population mobility by each Google `location' "
        f'({", ".join([i.replace("_percent_change_from_baseline", "").replace("_", " ") for i in mob_locs])}).\n\n'
    )
    tex_doc.add_line(description, section="Mobility extension", subsection="Data processing")
    non_wa_data = state_data.copy().loc[state_data["sub_region_1"] != "Western Australia"]

    # Add state population totals to dataframe
    state_pop_totals = load_raw_pop_data("31010do002_202206.xlsx").sum()
    for juris in jurisdictions:
        non_wa_data.loc[non_wa_data["sub_region_1"] == juris, "weights"] = state_pop_totals[juris]

    # Calculate weighted averages
    state_averages = pd.DataFrame(columns=mob_locs)
    for mob_loc in mob_locs:
        state_averages[mob_loc] = non_wa_data.groupby(non_wa_data.index).apply(
            lambda x: np.average(x[mob_loc], weights=x["weights"]),
        )
    return state_averages


def get_relative_mobility(
    mobility_df: pd.DataFrame,
) -> pd.DataFrame:
    """Google mobility values are provided as percentage change relative to baseline.
    However, we need mobility value relative to baseline estimate.

    Args:
        mobility_df: Data with values in original units

    Returns:
        Converted data
    """
    mobility_df.columns = [c.replace(CHANGE_STR, "") for c in mobility_df.columns]
    return 1.0 + mobility_df * 1e-2


def map_mobility_locations(
    wa_relmob: pd.DataFrame,
    non_wa_relmob: pd.DataFrame,
    tex_doc: StandardTexDoc,
    cross_ref: bool = True,
) -> pd.DataFrame:
    """
    Map mobility estimated values to model locations.

    Args:
        wa_relmob: Western Australia relative mobility
        non_wa_relmob: Rest of Australia relative mobility

    Returns:
        Mobility functions for use in the model
    """
    table_ref = (
        "the mapping algorithm displayed in Table \\ref{mob_map}"
        if cross_ref
        else "a mapping algorithm"
    )
    description = (
        f"Next, we used {table_ref} to map "
        "from Google's reported `locations' to the contact locations of the model's mixing matrix. "
    )
    tex_doc.add_line(description, section="Mobility extension", subsection="Data processing")

    patch_data = {"wa": wa_relmob, "non_wa": non_wa_relmob}
    model_mob = pd.DataFrame(
        columns=pd.MultiIndex.from_product([patch_data.keys(), MOBILITY_MAP.keys()])
    )
    for patch, p_data in patch_data.items():
        for mob_loc, mob_map in MOBILITY_MAP.items():
            data = p_data.assign(**mob_map).mul(p_data).sum(1)
            model_mob.loc[:, (patch, mob_loc)] = data
    return model_mob


def get_processed_mobility_data(
    tex_doc: StandardTexDoc,
    cross_ref: bool = True,
) -> pd.DataFrame:
    """Convert Google mobility data from raw form to form needed by model.

    Args:
        tex_doc: Documentation object
        cross_ref: Pass argument through to get_raw_state_mobility

    Returns:
        The processed mobility data
    """
    state_data, jurisdictions, mob_locs = get_raw_state_mobility(tex_doc, cross_ref)
    wa_data = state_data.loc[state_data["sub_region_1"] == "Western Australia", mob_locs]
    state_averages = get_non_wa_mob_averages(
        state_data, mob_locs, jurisdictions, tex_doc, cross_ref
    )

    description = (
        "Values were then converted from the reported percentage "
        "change from baseline to the proportional change relative to baseline, "
        "to obtain contact scaling factors. "
    )
    tex_doc.add_line(description, section="Mobility extension", subsection="Data processing")

    non_wa_relmob = get_relative_mobility(state_averages)
    wa_relmob = get_relative_mobility(wa_data)
    for location in MOBILITY_MAP:
        if sum(MOBILITY_MAP[location].values()) != 1.0:
            raise ValueError(f"Mobility mapping does not sum to one for {location}")
    processed_mob = map_mobility_locations(wa_relmob, non_wa_relmob, tex_doc, cross_ref)

    description = (
        f"Next, we took the {MOBILITY_AVERAGE_WINDOW}-day moving average to smooth the "
        "often abrupt shifts in mobility, including with weekend and public holidays. "
    )
    tex_doc.add_line(description, section="Mobility extension", subsection="Data processing")

    smoothed_mob = processed_mob.rolling(MOBILITY_AVERAGE_WINDOW).mean().dropna()

    fig_ref = (
        "These sequentially processed functions of time are illustrated in Figure \\ref{processed_mobility}. "
        if cross_ref
        else ""
    )
    description = (
        "Last, we squared the relative variations in mobility to account for "
        "the effect of reductions in visits to specific locations for both the infector "
        f"and the infectee of the modelled social contacts. {fig_ref}"
    )
    tex_doc.add_line(description, section="Mobility extension", subsection="Data processing")

    squared_mob = smoothed_mob**2.0
    return squared_mob


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
        interp_funcs[state][mob_loc] = linear_interp(
            epoch.dti_to_index(mob_values.index), mob_values[state, mob_loc].to_numpy()
        )
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
    working_matrix = matrices["home"] + matrices["school"]
    for location in ["other_locations", "work"]:
        for patch in ["wa", "non_wa"]:
            prop = wa_prop_func if patch == "wa" else 1.0 - wa_prop_func
            working_matrix += matrices[location] * mob_funcs[patch][location] * prop
    return working_matrix
