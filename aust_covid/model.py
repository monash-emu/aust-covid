import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'
from copy import copy
from jax import numpy as jnp

from summer2.functions.time import get_linear_interpolation_function as linear_interp
from summer2 import CompartmentalModel, Stratification, StrainStratification, Multiply
from summer2.parameters import Parameter, Function, Time

from aust_covid.utils import triangle_wave_func, add_image_to_doc
from aust_covid.inputs import load_pop_data, load_uk_pop_data, get_base_vacc_data
from aust_covid.tracking import track_incidence, track_notifications, track_deaths, track_adult_seroprev, track_strain_prop, track_reproduction_number, track_immune_prop
from aust_covid.mobility import get_processed_mobility_data, get_interp_funcs_from_mobility, get_dynamic_matrix
from aust_covid.vaccination import add_derived_data_to_vacc, calc_vacc_funcs_from_props, get_model_vacc_vals_from_data
from emutools.tex import StandardTexDoc
from emutools.parameters import capture_kwargs
from inputs.constants import REFERENCE_DATE, ANALYSIS_START_DATE, ANALYSIS_END_DATE, WA_REOPEN_DATE, MATRIX_LOCATIONS
from inputs.constants import N_LATENT_COMPARTMENTS, AGE_STRATA, STRAIN_STRATA, INFECTION_PROCESSES, IMMUNITY_STRATA
from inputs.constants import SUPPLEMENT_PATH, DATA_PATH


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""


def build_model(
    tex_doc: StandardTexDoc,
    mobility_sens: bool=False,
    vacc_sens: bool=False,
):

    # Model construction
    n_infectious_comps = N_LATENT_COMPARTMENTS
    latent_compartments = [f'latent_{i}' for i in range(N_LATENT_COMPARTMENTS)]
    infectious_compartments = [f'infectious_{i}' for i in range(n_infectious_comps)]
    compartments = ['susceptible', 'recovered', 'waned'] + infectious_compartments + latent_compartments

    aust_model = build_base_model(compartments, infectious_compartments, tex_doc)
    epoch = aust_model.get_epoch()
    model_pops = load_pop_data(tex_doc)
    set_starting_conditions(aust_model, model_pops, tex_doc)
    add_infection(aust_model, latent_compartments, tex_doc)
    add_latent_transition(aust_model, latent_compartments, infectious_compartments, tex_doc)
    add_infectious_transition(aust_model, infectious_compartments, tex_doc)
    add_waning(aust_model, tex_doc)

    # Age and heterogeneous mixing
    wa_reopen_func = get_wa_infection_scaling(aust_model)
    raw_matrices = {l: pd.read_csv(DATA_PATH / f'{l}.csv', index_col=0).to_numpy() for l in MATRIX_LOCATIONS}
    adjusted_matrices = adapt_gb_matrices_to_aust(raw_matrices, model_pops, tex_doc)

    # Mobility effects
    if mobility_sens:
        state_props = model_pops.sum() / model_pops.sum().sum()
        model_mob = get_processed_mobility_data(tex_doc)
        interp_funcs = get_interp_funcs_from_mobility(model_mob, epoch)
        wa_reopen_func = get_wa_infection_scaling(aust_model)
        wa_prop_func = wa_reopen_func * state_props['wa']
        wa_funcs = Function(capture_kwargs, kwargs=interp_funcs['wa'])
        non_wa_funcs = Function(capture_kwargs, kwargs=interp_funcs['non_wa'])
        mob_funcs = Function(capture_kwargs, kwargs={'wa': wa_funcs, 'non_wa': non_wa_funcs})
        mixing_matrix = Function(get_dynamic_matrix, [adjusted_matrices, mob_funcs, wa_prop_func])
    else:
        mixing_matrix = sum(list(adjusted_matrices.values()))

    age_strat = get_age_stratification(compartments, mixing_matrix, tex_doc)
    aust_model.stratify_with(age_strat)

    # Other stratifications and reinfection
    strain_strat = get_strain_stratification(compartments, tex_doc)
    aust_model.stratify_with(strain_strat)
    seed_vocs(aust_model, latent_compartments, tex_doc)

    add_reinfection(aust_model, latent_compartments, tex_doc)

    spatial_strat = get_spatial_stratification(compartments, model_pops, tex_doc, wa_reopen_func)
    aust_model.stratify_with(spatial_strat)

    if vacc_sens:
        imm_strat = get_vacc_imm_strat(compartments, tex_doc)
        aust_model.stratify_with(imm_strat)

        vacc_df = get_base_vacc_data()
        _, ext_vacc_df = add_derived_data_to_vacc(vacc_df)
        primary_rates = ext_vacc_df['rate primary full'].dropna()
        boost_rates = ext_vacc_df['rate adult booster'].dropna()
        primary_func = linear_interp(epoch.dti_to_index(primary_rates.index), primary_rates)
        boost_func = linear_interp(epoch.dti_to_index(boost_rates.index), boost_rates)

        for comp in aust_model._original_compartment_names:
            aust_model.add_transition_flow(
                'vaccination',
                primary_func,
                source=comp.name,
                dest=comp.name,
                source_strata={'immunity': 'unvacc', 'agegroup': '5'},
                dest_strata={'immunity': 'vacc', 'agegroup': '5'},
            )
            for age_strat in AGE_STRATA[3:]:
                aust_model.add_transition_flow(
                    'vaccination',
                    boost_func,
                    source=comp.name,
                    dest=comp.name,
                    source_strata={'immunity': 'unvacc', 'agegroup': str(age_strat)},
                    dest_strata={'immunity': 'vacc', 'agegroup': str(age_strat)},
                )
        
            aust_model.add_transition_flow(
                'waning',
                1.0 / Parameter('vacc_immune_period'),
                source=comp.name,
                dest=comp.name,
                source_strata={'immunity': 'vacc'},
                dest_strata={'immunity': 'waned'}
            )

    else:
        imm_strat = get_default_imm_strat(compartments, tex_doc)
        aust_model.stratify_with(imm_strat)

    # Outputs
    track_incidence(aust_model, tex_doc)
    track_notifications(aust_model, tex_doc)
    track_deaths(aust_model, tex_doc)
    track_adult_seroprev(compartments, aust_model, 15, tex_doc)
    track_strain_prop(aust_model, infectious_compartments, tex_doc)
    track_immune_prop(aust_model)
    track_reproduction_number(aust_model, infectious_compartments, tex_doc)

    for comp in compartments:
        aust_model.request_output_for_compartments(comp, [comp])

    # Compartment initialisation
    initialise_comps(model_pops, aust_model, vacc_sens, tex_doc)

    return aust_model


def build_base_model(
    compartments: list,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = f'The base model consists of {len(compartments)} states, ' \
        f'representing the following states: {", ".join(compartments).replace("_", "")}. ' \
        f"Each of the infectious compartments contribute equally to the force of infection. \n"
    time_desc =  f'A simulation is run from {ANALYSIS_START_DATE.strftime("%d %B %Y")} to {ANALYSIS_END_DATE.strftime("%d %B %Y")}. '
    tex_doc.add_line(description, 'Model Structure')
    tex_doc.add_line(time_desc, 'Population')

    return CompartmentalModel(
        times=(
            (ANALYSIS_START_DATE - REFERENCE_DATE).days, 
            (ANALYSIS_END_DATE - REFERENCE_DATE).days,
        ),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        ref_date=REFERENCE_DATE,
    )


def set_starting_conditions(
    model: CompartmentalModel,
    pop_data: pd.DataFrame,
    tex_doc: StandardTexDoc,
) -> str:
    total_pop = pop_data.sum().sum()
    description = f'The simulation starts with {str(round(total_pop / 1e6, 3))} million fully susceptible persons, ' \
        'with the infection process triggered through strain seeding as described below. '
    tex_doc.add_line(description, 'Population')

    model.set_initial_population({'susceptible': total_pop})


def add_infection(
    model: CompartmentalModel,
    latent_compartments: list,
    tex_doc: StandardTexDoc,
) -> str:
    process = 'infection'
    origin = 'susceptible'
    destination = latent_compartments[0]
    description = f'The {process} process moves people from the {origin.replace("_", "")} ' \
        f'compartment to the {destination.replace("_", "")} compartment ' \
        '(being the first latent compartment), ' \
        'under the frequency-dependent transmission assumption. '
    tex_doc.add_line(description, 'Model Structure')

    model.add_infection_frequency_flow(process, Parameter('contact_rate'), origin, destination)


def add_latent_transition(
    model: CompartmentalModel,
    latent_compartments: list,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
):
    parameter_name = 'latent_period'
    final_dest = infectious_compartments[0]
    description = f'Following infection, infected persons enter a series of {N_LATENT_COMPARTMENTS} latent compartments. ' \
        'These are chained in sequence, with infected persons transitioning sequentially from ' \
        f'compartment 0 through to compartment {len(latent_compartments) - 1}. ' \
        'To achieve the same mean sojourn time in the composite latent stage, ' \
        'the rate of transition between successive latent compartments and out of the last latent compartment ' \
        f'are multiplied by the number of serial compartments (i.e. {N_LATENT_COMPARTMENTS}). ' \
        'As persons exit the final latent compartment, they enter the first infectious compartment. ' \
        'An Erlang-distributed infectious and latent duration is consistent with epidemiological evidence ' \
        'and our intuition around this quantity. The serial interval \cite{anderheiden2022} and generation time \cite{ito2022} appear to ' \
        'well represented by a gamma distribution, with multiple past modelling studies choosing ' \
        'a shape parameter of four or five having been previously used to fit this distribution \cite{davies2020b,davies2020c}. '
    tex_doc.add_line(description, 'Model Structure')

    rate = 1.0 / Parameter(parameter_name) * N_LATENT_COMPARTMENTS
    for i_comp in range(N_LATENT_COMPARTMENTS - 1):
        origin = latent_compartments[i_comp]
        destination = latent_compartments[i_comp + 1]
        model.add_transition_flow(f'latent_transition_{str(i_comp)}', rate, origin, destination)
    model.add_transition_flow('progression', rate, latent_compartments[-1], final_dest)


def add_infectious_transition(
    model: CompartmentalModel,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
):
    parameter_name = 'infectious_period'
    final_dest = 'recovered'
    n_inf_comps = len(infectious_compartments)
    description = f'As for the latent compartments, the infectious compartments are also chained in series, ' \
        f'with a total of {n_inf_comps} also chained together in sequence. ' \
        'As for the latent compartments, ' \
        f'each transition rate is multiplied by {n_inf_comps}. ' \
        'As persons exit the final infectious compartment, they enter the recovered compartment.\n'    
    tex_doc.add_line(description, 'Model Structure')

    rate = 1.0 / Parameter(parameter_name) * n_inf_comps
    for i_comp in range(n_inf_comps - 1):
        origin = infectious_compartments[i_comp]
        destination = infectious_compartments[i_comp + 1]
        model.add_transition_flow(f'inf_transition_{str(i_comp)}', rate, origin, destination)
    model.add_transition_flow('recovery', rate, infectious_compartments[-1], final_dest)


def add_waning(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
) -> str:
    process = 'waning'
    origin = 'recovered'
    destination = 'waned'
    parameter_name = 'natural_immunity_period'
    description = 'A waned compartment is included in the model ' \
        'to represent persons who no longer have natural immunity from past SARS-CoV-2 infection. ' \
        f'As these persons lose their infection-induced immunity, they transition from the ' \
        f'{origin.replace("_", "")} compartment to the {destination.replace("_", "")} compartment '\
        f'at a rate equal to the reciprocal of the {parameter_name.replace("_", " ")} parameter. '
    tex_doc.add_line(description, 'Model Structure')

    model.add_transition_flow(process, 1.0 / Parameter(parameter_name), origin, destination)


def plot_mixing_matrices(
    matrices: dict, 
    strata: list, 
    filename: str,
    tex_doc: StandardTexDoc,
) -> tuple:
    matrix_figsize = 800
    matrix_fig = make_subplots(rows=2, cols=2, subplot_titles=MATRIX_LOCATIONS)
    positions = [[1, 1], [1, 2], [2, 1], [2, 2]]
    for i_loc, loc in enumerate(MATRIX_LOCATIONS):
        cur_pos = positions[i_loc]
        matrix_fig.add_trace(go.Heatmap(x=strata, y=strata, z=matrices[loc], coloraxis = 'coloraxis'), cur_pos[0], cur_pos[1])
    matrix_fig.update_layout(width=matrix_figsize, height=matrix_figsize * 1.15)

    caption = f'Daily contact rates by age group (row), contact age group (column) and location (panel). '
    add_image_to_doc(matrix_fig, 'input_population', caption, tex_doc, 'Mixing')
    return matrix_fig


def adapt_gb_matrices_to_aust(
    unadjusted_matrices: dict, 
    pop_data: pd.DataFrame,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = 'Social contact matrices for Great Britain ' \
        'were adjusted to account for the differences in the age distribution betweem the ' \
        'Australian population distribution in 2022 and the population of Great Britain in 2000. ' \
        'The matrices were adjusted by taking the dot product of the location-specific unadjusted matrices and the diagonal matrix ' \
        'containing the vector of the ratios between the proportion of the British and Australian populations ' \
        'within each age bracket as its diagonal elements. '
    tex_doc.add_line(description, 'Mixing')

    # Australia population
    aust_props_disp = copy(pop_data)
    aust_props_disp['age_group'] = [f'{age}-{age + 4}' for age in AGE_STRATA[:-1]] + ['75 and over']

    input_pop_fig = px.bar(
        aust_props_disp.melt(id_vars=['age_group']), 
        x='age_group', 
        y='value', 
        color='variable', 
        labels={'value': 'population', 'age_group': ''},
    )
    caption = 'Australian population sizes implemented in the model obtained from Australia Bureau of Statistics.'
    add_image_to_doc(input_pop_fig, 'input_population', caption, tex_doc, 'Mixing')

    # UK population
    raw_uk_data = load_uk_pop_data(tex_doc)
    uk_pop_fig = px.bar(raw_uk_data)
    uk_pop_fig.update_layout(showlegend=False)
    caption = 'United Kingdom population sizes used in matrix weighting.'
    add_image_to_doc(input_pop_fig, 'uk_population', '', tex_doc, caption)

    # Weighting calculations
    aust_age_props = pop_data.sum(axis=1) / pop_data.sum().sum()
    uk_age_pops = raw_uk_data[:15]
    uk_age_pops['75 or over'] = raw_uk_data[15:].sum()
    uk_age_pops.index = AGE_STRATA
    uk_age_props = uk_age_pops / uk_age_pops.sum()
    aust_uk_ratios = aust_age_props / uk_age_props

    # Adjust each location-specific matrix
    adjusted_matrices = {}
    for location in MATRIX_LOCATIONS:
        unadjusted_matrix = unadjusted_matrices[location]
        assert unadjusted_matrix.shape[0] == unadjusted_matrix.shape[1], 'Unadjusted mixing matrix not square'
        assert len(aust_age_props) == unadjusted_matrix.shape[0], 'Different number of Aust age groups from mixing categories'
        assert len(uk_age_props) == unadjusted_matrix.shape[0], 'Different number of UK age groups from mixing categories'
        adjusted_matrices[location] = np.dot(unadjusted_matrix, np.diag(aust_uk_ratios))
    
    # Plot matrices
    raw_matrix_fig = plot_mixing_matrices(unadjusted_matrices, AGE_STRATA, 'raw_matrices', tex_doc)
    adj_matrix_fig = plot_mixing_matrices(adjusted_matrices, AGE_STRATA, 'adjusted_matrices', tex_doc)
    add_image_to_doc(raw_matrix_fig, 'raw_matrices', '', tex_doc, 'Raw mixing matrices')
    add_image_to_doc(adj_matrix_fig, 'adjusted_matrices', '', tex_doc, 'Adjusted mixing matrices')

    return adjusted_matrices


def get_age_stratification(
    compartments: list,
    matrix: np.array,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = 'We stratified all compartments of the model described into sequential age brackets cmprising 5-year ' \
        f'bands from age {AGE_STRATA[0]} to {AGE_STRATA[0] + 4} through to age {AGE_STRATA[-2]} to {AGE_STRATA[-2] + 4}, ' \
        f'with a final age band to represent those aged {AGE_STRATA[-1]} and above. ' \
        'These age brackets were chosen to match those used by the POLYMOD survey \cite{mossong2008} ' \
        'and so fit with the mixing data available. ' \
        'The population distribution by age group was informed by the data from the Australian ' \
        'Bureau of Statistics introduced previously. ' \
        'Ageing between sequential bands was not permitted given the time window of the simulation. '
    tex_doc.add_line(description, 'Stratification', subsection='Age')

    age_strat = Stratification('agegroup', AGE_STRATA, compartments)
    age_strat.set_mixing_matrix(matrix)
    return age_strat


def get_strain_stratification(
    compartments: list,
    tex_doc: StandardTexDoc,
) -> tuple:
    strain_strings = [f'{strain.replace("ba", "BA.")}' for strain in STRAIN_STRATA]
    compartments_to_stratify = [comp for comp in compartments if comp != 'susceptible']
    description = f'We stratified the following compartments according to strain: ' \
        f'{", ".join(compartments_to_stratify).replace("_", "")}, ' \
        'replicating all of these compartments to represent the various major Omicron sub-variants relevant to the 2022 epidemic, ' \
        f'namely: {", ".join(strain_strings)}. ' \
        f"This was implemented using the summer library's `{StrainStratification.__name__}' class. "
    tex_doc.add_line(description, 'Stratification', subsection='Omicron Sub-variants')

    return StrainStratification('strain', STRAIN_STRATA, compartments_to_stratify)


def get_default_imm_strat(
    compartments: list, 
    tex_doc: StandardTexDoc,
) -> Stratification:
    description = 'All compartments and stratifications described were further ' \
        'stratified into two strata with differing levels of susceptibility to infection. ' \
        'One parameter was used to represent the proportion of the population with ' \
        'immunological protection against infection, ' \
        'with a second parameter used to quantify the relative reduction in ' \
        'the rate of infection and reinfection for those in the stratum with ' \
        'reduced susceptibility. ' \
        'This stratification was implemented because some heterogeneity in susceptibility ' \
        'may have been introduced through a proportion of the population having greater ' \
        'protection through vaccination (e.g. because of recent receipt of a booster dose) ' \
        'or other immunological heterogeneity in the population, ' \
        'and because earlier iterations of the model suggested this helped to capture ' \
        'sharpness of the peak of the initial BA.1 wave. '
    tex_doc.add_line(description, 'Stratification', subsection='Heterogeneous susceptibility')

    imm_strat = Stratification('immunity', ['imm', 'nonimm'], compartments)
    for infection_process in INFECTION_PROCESSES:
        heterogeneity = {'imm': Multiply(1.0 - Parameter('imm_infect_protect')), 'nonimm': None}
        imm_strat.set_flow_adjustments(infection_process, heterogeneity)
    return imm_strat


def get_vacc_imm_strat(
    compartments: list, 
    tex_doc: StandardTexDoc,
) -> Stratification:

    imm_strat = Stratification('immunity', ['unvacc', 'vacc', 'waned'], compartments)
    for infection_process in INFECTION_PROCESSES:
        heterogeneity = {'unvacc': None, 'vacc': Multiply(1.0 - Parameter('imm_infect_protect')), 'waned': None}
        imm_strat.set_flow_adjustments(infection_process, heterogeneity)
    return imm_strat


def seed_vocs(
    model: CompartmentalModel,
    latent_compartments: list,
    tex_doc: StandardTexDoc,
) -> str:
    strains = model.stratifications['strain'].strata
    seed_comp = latent_compartments[0]
    seed_duration_str = 'seed_duration'
    seed_rate_str = 'seed_rate'
    description = f'Each strain (including the starting {strains[0].replace("ba", "BA.")} strain) was seeded through ' \
        'a triangular step function that introduces new infectious ' \
        f'persons into the {seed_comp.replace("_", "")} compartment over a fixed seeding duration defined by a single ' \
        f'{seed_duration_str.replace("_", " ")} parameter. ' \
        f'and at a peak rate defined by one {seed_rate_str.replace("_", " ")} parameter. ' \
        'The time of first emergence of each strain into the system is defined by ' \
        'a separate emergence time parameter for each strain. '
    tex_doc.add_line(description, 'Stratification', subsection='Omicron Sub-variants')

    for strain in strains:
        seed_args = [Time, Parameter(f'{strain}_seed_time'), Parameter(seed_duration_str), Parameter(seed_rate_str)]
        voc_seed_func = Function(triangle_wave_func, seed_args)
        model.add_importation_flow(f'seed_{strain}', voc_seed_func, seed_comp, dest_strata={'strain': strain}, split_imports=True)


def add_reinfection(
    model: CompartmentalModel,
    latent_compartments: list,
    tex_doc: StandardTexDoc,
) -> str:
    destination = latent_compartments[0]
    description = 'Reinfection is possible from both the recovered ' \
        'and waned compartments, which we refer to as  ' \
        "`early' and `late' reinfection respectively. " \
        'In the case of early reinfection, this is only possible ' \
        'for persons who have recovered from an earlier circulating sub-variant. ' \
        'That is, early BA.2 reinfection is possible for persons previously infected with ' \
        'BA.1, while early BA.5 reinfection is possible for persons previously infected with ' \
        'BA.1 or BA.2. The parameter governing the degree of immune escape is determined ' \
        'by the infecting variant and differs for BA.2 and BA.5. ' \
        'Therefore, the rate of reinfection is equal for BA.5 reinfecting those recovered from past BA.1 infection ' \
        'as for those recovered from past BA.2 infection. ' \
        'For late reinfection, all natural immunity is lost for persons in the waned compartment, ' \
        'such that the rate of reinfection for these persons is the same as the rate of infection ' \
        'for fully susceptible persons. ' \
        'As for the process of first infection, all reinfection processes transition individuals ' \
        'to the latent compartment corresponding to the infecting strain.\n'
    tex_doc.add_line(description, 'Reinfection')

    for dest_strain in STRAIN_STRATA:
        for source_strain in STRAIN_STRATA:
            escape = Parameter(f'{dest_strain}_escape') if int(dest_strain[-1]) > int(source_strain[-1]) else 0.0
            model.add_infection_frequency_flow(
                'early_reinfection', 
                Parameter('contact_rate') * escape,
                'recovered',
                destination,
                source_strata={'strain': source_strain},
                dest_strata={'strain': dest_strain},
            )
            model.add_infection_frequency_flow(
                'late_reinfection', 
                Parameter('contact_rate'),
                'waned', 
                destination,
                source_strata={'strain': source_strain},
                dest_strata={'strain': dest_strain},
            )


def get_wa_infection_scaling(
    model: CompartmentalModel,
) -> Function:
    reopen_param_str = 'wa_reopen_period'
    reopen_index = model.get_epoch().dti_to_index(WA_REOPEN_DATE)
    reopen_times = [reopen_index, reopen_index + Parameter(reopen_param_str)]
    return linear_interp(reopen_times, np.array([0.0, 1.0]))


def get_spatial_stratification(
    compartments: list, 
    model_pops: pd.DataFrame, 
    tex_doc: StandardTexDoc,
    reopen_func,
) -> Stratification:
    strata = model_pops.columns
    reopen_param_str = 'wa_reopen_period'
    description = 'All model compartments previously described are further ' \
        f"stratified into strata to represent Western Australia ({strata[0].upper()}) and `{strata[1]}' " \
        'to represent the remaining major jurisdictions of Australia. ' \
        f'Transmission in {strata[0].upper()} was initially set to zero, ' \
        f'and subsquently scaled up to being equal to that of the {strata[1]} ' \
        f"jurisdictions of Australia over a period that governed by the `{reopen_param_str.replace('_', ' ')}' parameter. "
    tex_doc.add_line(description, 'Stratification', subsection='Spatial')

    spatial_strat = Stratification('states', model_pops.columns, compartments)
    infection_adj = {strata[0]: reopen_func, strata[1]: None}
    for infection_process in INFECTION_PROCESSES:
        spatial_strat.set_flow_adjustments(infection_process, infection_adj)
    return spatial_strat


def initialise_comps(
    model_pops: pd.DataFrame, 
    model: CompartmentalModel, 
    vacc_sens: bool, 
    tex_doc: StandardTexDoc,
):
    """
    See "description" string below.

    Args:
        model_pops: Patch and age-specific population
        model: The epidemiological model
        start_props: Starting proportions in the vaccination analysis
        vacc_sens: Whether the vaccination analysis being run
        tex_doc: Documentation object
    """
    start_comp = 'susceptible'
    imm_prop_param = 'imm_prop'
    immunity_strata = model.stratifications['immunity'].strata
    imm_prop_str = imm_prop_param.replace('_', '\_')
    description = f'Starting model populations were distributed to the {start_comp} compartment by ' \
        f'age and spatial status ({model_pops.columns[0].upper()}, {model_pops.columns[1]}) ' \
        'according to the age distribution in each of the two simulated regions. ' \
        'These populations were then split by immunity status. ' \
        f'In the base case analysis, the proportion was set according to the {imm_prop_str} parameter. ' \
        'For the vaccination analysis, the starting immune population was set according to the ' \
        'first value for time-varying proportion recently boosted/vaccinated. '
    tex_doc.add_line(description, 'Initialisation')

    def get_init_pop(imm_prop):
        if vacc_sens:
            imm_props = {
                'unvacc': 1.0,
                'vacc': 0.0,
                'waned': 0.0,
            }
        else:
            imm_props = {
                'imm': imm_prop,
                'nonimm': 1.0 - imm_prop,
            }

        init_pop = jnp.zeros(len(model.compartments), dtype=np.float64)
        for age in AGE_STRATA:
            for state in model_pops:
                pop = model_pops.loc[age, state]
                for imm_status in immunity_strata:
                    comp_filter = {'name': start_comp, 'agegroup': str(age), 'states': state, 'immunity': imm_status}
                    query = model.query_compartments(comp_filter, as_idx=True)
                    init_pop = init_pop.at[query].set(pop * imm_props[imm_status])
        return init_pop

    model.init_population_with_graphobject(Function(get_init_pop, [Parameter(imm_prop_param)]))
