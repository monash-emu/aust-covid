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
from aust_covid.vaccination import add_derived_data_to_vacc
from emutools.tex import StandardTexDoc, get_tex_formatted_date
from emutools.parameters import capture_kwargs
from inputs.constants import REFERENCE_DATE, ANALYSIS_START_DATE, ANALYSIS_END_DATE, WA_REOPEN_DATE, MATRIX_LOCATIONS
from inputs.constants import N_LATENT_COMPARTMENTS, AGE_STRATA, STRAIN_STRATA, INFECTION_PROCESSES, IMMUNITY_STRATA
from inputs.constants import SUPPLEMENT_PATH, DATA_PATH


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""

REOPEN_PARAM_STR = 'wa_reopen_period'


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
    initialise_comps(model_pops, aust_model, vacc_sens)

    return aust_model


def build_base_model(
    compartments: list,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
) -> tuple:
    compartment_names = ", ".join(compartments).replace("_", "\_")
    description = f'The base model consists of {len(compartments)} states, ' \
        'representing sequential epidemiological conditions with regards SARS-CoV-2 infection and COVID-19 disease ' \
        f'({compartment_names}). ' \
        f"Each of the sequentially numbered infectious compartments contribute equally to the force of infection. \n"
    time_desc =  f'Each simulation is run from {get_tex_formatted_date(ANALYSIS_START_DATE)} to ' \
        f'{get_tex_formatted_date(ANALYSIS_END_DATE)}. '
    tex_doc.add_line(description, 'Base compartmental structure')
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
        'with the infection process triggered through subsequent strain seeding as described below. '
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
    dest_str = destination.replace('_', '\_')
    description = f'The {process} process moves people from the {origin} ' \
        f'compartment to the {dest_str} compartment ' \
        '(i.e. the first latent compartment), ' \
        'under the assumption of frequency-dependent transmission. '
    tex_doc.add_line(description, 'Base compartmental structure')

    model.add_infection_frequency_flow(process, Parameter('contact_rate'), origin, destination)


def add_latent_transition(
    model: CompartmentalModel,
    latent_compartments: list,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
):
    parameter_name = 'latent_period'
    final_dest = infectious_compartments[0]
    description = f'Following infection, infected persons proceed to transition through ' \
        f'a series of {N_LATENT_COMPARTMENTS} latent compartments. ' \
        'These are chained in sequence, with infected persons transitioning sequentially from ' \
        f'compartment 0 through to compartment {len(latent_compartments) - 1}. ' \
        'To achieve the same mean sojourn time in the composite latent stage, ' \
        'the rate of transition between successive latent compartments ' \
        'and of exiting the the last latent compartment ' \
        f'are multiplied by the number of serial compartments (i.e. {N_LATENT_COMPARTMENTS}). ' \
        'As persons exit the final latent compartment, they enter the first infectious compartment. ' \
        'An Erlang-distributed infectious and latent duration is consistent with epidemiological evidence ' \
        'that the serial interval \cite{anderheiden2022} and generation time \cite{ito2022} are often ' \
        'well represented by a gamma distribution, with multiple past modelling studies choosing ' \
        'a shape parameter of four or five having been previously used to fit this distribution \cite{davies2020b,davies2020c}. '
    tex_doc.add_line(description, 'Base compartmental structure')

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
        f'with a total of {n_inf_comps} again chained together in sequence. ' \
        f'As for the latent compartments, each transition rate is multiplied by {n_inf_comps}.\n ' \
        'As persons exit the final infectious compartment, they enter the recovered compartment. '    
    tex_doc.add_line(description, 'Base compartmental structure')

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
    tex_doc.add_line(description, 'Base compartmental structure')

    model.add_transition_flow(process, 1.0 / Parameter(parameter_name), origin, destination)


def plot_mixing_matrices(
    matrices: dict, 
    strata: list, 
) -> tuple:
    matrix_figsize = 800
    matrix_fig = make_subplots(rows=2, cols=2, subplot_titles=[m.replace('_', ' ') for m in MATRIX_LOCATIONS], vertical_spacing=0.08, horizontal_spacing=0.07)
    positions = [[1, 1], [1, 2], [2, 1], [2, 2]]
    for i_loc, loc in enumerate(MATRIX_LOCATIONS):
        cur_pos = positions[i_loc]
        matrix_fig.add_trace(go.Heatmap(x=strata, y=strata, z=matrices[loc], coloraxis = 'coloraxis'), cur_pos[0], cur_pos[1])
    return matrix_fig.update_layout(width=matrix_figsize, height=matrix_figsize * 0.9)


def adapt_gb_matrices_to_aust(
    unadjusted_matrices: dict, 
    pop_data: pd.DataFrame,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = 'Raw, location-specific social contact matrices from the POLYMOD study ' \
        'for Great Britain (Figure \\ref{raw_matrices}) ' \
        'were adjusted to account for the differences in the age distribution betweem the ' \
        'Australian population distribution in 2022 (Figure \\ref{input_population}) ' \
        'and the population of Great Britain in 2000 (Figure \\ref{uk_population}). ' \
        'The raw matrices were adjusted by taking the dot product of the location-specific unadjusted matrices ' \
        'and the diagonal matrix containing the vector of the ratios between ' \
        'the proportion of the British and Australian populations ' \
        'within each age bracket as its diagonal elements. ' \
        'In analyses without contact scaling for mobility, the resulting adjusted matrices ' \
        'summed over location (Figure \\ref{adjusted_matrices}) were implemented ' \
        'as fixed rates of contact between each possible pair of age groups. '
    tex_doc.add_line(description, 'Mixing')

    # Australia population
    aust_props_disp = copy(pop_data)
    aust_props_disp['age_group'] = [f'{age}-{age + 4}' for age in AGE_STRATA[:-1]] + ['75 and over']

    pop_labels = {'value': 'population', 'age_group': 'age group'}
    input_pop_data = aust_props_disp.melt(id_vars=['age_group'])
    input_pop_fig = px.bar(input_pop_data, x='age_group', y='value', color='variable', labels=pop_labels)
    input_pop_fig = input_pop_fig.update_layout(height=400, showlegend=False)
    title = 'Stacked Australian population sizes implemented in the model.'
    caption = 'Western Australia (blue bars), aggregate of remaining major jurisdictions of Australia (red bars).'
    add_image_to_doc(input_pop_fig, 'input_population', 'svg', title, tex_doc, 'Mixing', caption=caption)

    # UK population
    raw_uk_data = load_uk_pop_data(tex_doc)
    uk_pop_fig = px.bar(raw_uk_data, labels=pop_labels)
    uk_pop_fig.update_layout(showlegend=False, height=400)
    caption = 'United Kingdom population sizes used in matrix weighting.'
    add_image_to_doc(uk_pop_fig, 'uk_population', 'svg', caption, tex_doc, 'Mixing')

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
    raw_matrix_fig = plot_mixing_matrices(unadjusted_matrices, AGE_STRATA)
    adj_matrix_fig = plot_mixing_matrices(adjusted_matrices, AGE_STRATA)
    caption = 'Number of contacts per day by respondent age group (row), contact age group (column) and location (panel). '
    title = 'Raw contact rates obtained from POLYMOD surveys for the United Kingdom.'
    add_image_to_doc(raw_matrix_fig, 'raw_matrices', 'svg', title, tex_doc, 'Mixing', caption=caption)
    title = 'Contact rates after adjustment to the Australian population distribution.'
    add_image_to_doc(adj_matrix_fig, 'adjusted_matrices', 'svg', title, tex_doc, 'Mixing', caption=caption)

    return adjusted_matrices


def get_age_stratification(
    compartments: list,
    matrix: np.array,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = 'We stratified all compartments of the base model described above ' \
        '(Section \\ref{base_compartmental_structure}) into sequential age brackets comprising 5-year ' \
        f'bands from age {AGE_STRATA[0]} to {AGE_STRATA[0] + 4} through to age {AGE_STRATA[-2]} to {AGE_STRATA[-2] + 4}, ' \
        f'with a final age band to represent those aged {AGE_STRATA[-1]} and above. ' \
        'These age brackets were chosen to match those used by the POLYMOD survey \cite{mossong2008} ' \
        'and so fit with the mixing approach implemented (described below in Section \\ref{mixing}). ' \
        'The population distribution by modelled age group was obtained from the Australian ' \
        'Bureau of Statistics data introduced previously (Section \\ref{population}, ' \
        'Figure \\ref{input_population}). ' \
        'Ageing between sequential bands was not permitted given the short time window of the simulation. '
    tex_doc.add_line(description, 'Stratification', subsection='Age')

    age_strat = Stratification('agegroup', AGE_STRATA, compartments)
    age_strat.set_mixing_matrix(matrix)
    return age_strat


def get_strain_stratification(
    compartments: list,
    tex_doc: StandardTexDoc,
) -> tuple:
    strain_strings = [f'{strain.replace("ba", "BA.")}' for strain in STRAIN_STRATA]
    non_strat_comp = 'susceptible'
    compartments_to_stratify = [comp for comp in compartments if comp != non_strat_comp]
    comps_to_stratify_str = ", ".join(compartments_to_stratify).replace('_', '\_')
    description = f'We stratified all compartments other than {non_strat_comp} according to strain ' \
        f'({comps_to_stratify_str}), ' \
        'replicating all of these compartments to represent the various major Omicron sub-variants ' \
        "relevant to Australia's 2022 epidemic, " \
        f'namely: {", ".join(strain_strings)}. ' \
        f"This was implemented using the summer library's `{StrainStratification.__name__}' class.\n"
    # TODO: Is it appropriate to link to summer2 repo code
    tex_doc.add_line(description, 'Stratification', subsection='Omicron Sub-variants')

    return StrainStratification('strain', STRAIN_STRATA, compartments_to_stratify)


def get_default_imm_strat(
    compartments: list, 
    tex_doc: StandardTexDoc,
) -> Stratification:
    imm_strata = ['imm', 'nonimm']
    protect_param = 'imm_infect_protect'
    protect_param_str = protect_param.replace('_', '\_')
    description = 'All (multiply stratified) compartments introduced above were further ' \
        f'stratified into {len(imm_strata)} strata with differing levels of susceptibility to infection. ' \
        f'A calibrated parameter ({protect_param_str}) was used to quantify the relative reduction in the rate of ' \
        'infection and reinfection for those in the stratum with reduced susceptibility. ' \
        'In the two analyses without extension for vaccination, ' \
        'a further parameter was calibrated to represent the proportion of the population with ' \
        'immunological protection against infection. ' \
        'This approach was adopted because some population heterogeneity in susceptibility ' \
        'may have been introduced through a proportion of the population having greater ' \
        'protection through vaccination, boosting or other intrinsic individual variation that ' \
        'would not otherwise be captured under the assumptions inherent in our compartmental model. ' \
        'By contrast to the vaccination extension approach, no flows between these strata were applied, ' \
        'such that the proportion of the population in each of the two strata remains fixed over time. '
    tex_doc.add_line(description, 'Stratification', subsection='Heterogeneous susceptibility')

    imm_strat = Stratification('immunity', imm_strata, compartments)
    for infection_process in INFECTION_PROCESSES:
        heterogeneity = {'imm': Multiply(1.0 - Parameter(protect_param)), 'nonimm': None}
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
        f'variable {seed_duration_str.replace("_", " ")} parameter (i.e. applied to all subvariants) ' \
        f'at a peak rate defined by a single {seed_rate_str.replace("_", " ")} parameter ' \
        '(also applied to all subvariants). ' \
        'The time of first emergence of each strain into the system was defined by ' \
        'a separate emergence time parameter for each modelled subvariant strain. '
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
    description = 'We modelled reinfection from both the recovered ' \
        "and waned compartments, which we term `early' and `late' reinfection respectively. " \
        'In the case of early reinfection, this is only possible ' \
        'for persons who have recovered from an earlier circulating sub-variant. ' \
        'That is, early BA.2 reinfection is possible for persons previously infected with ' \
        'BA.1, and early BA.5 reinfection is possible for persons previously infected with ' \
        'BA.1 or BA.2, while other reinfection processes are not permitted.\n' \
        'The parameter governing the degree of immune escape is determined ' \
        'by the infecting variant was estimated separately for BA.2 and BA.5. ' \
        'Therefore, the rate of reinfection is equal for BA.5 reinfecting those recovered from past BA.1 infection ' \
        'as for those recovered from past BA.2 infection.\n\n ' \
        'For late reinfection, all natural immunity is lost for persons in the waned compartment, ' \
        'such that the rate of reinfection for these persons is the same as the rate of infection ' \
        'for fully susceptible persons. \n' \
        'As for the process of first infection, all reinfection processes transition individuals ' \
        'to the latent compartment corresponding to the infecting strain.'
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
    reopen_index = model.get_epoch().dti_to_index(WA_REOPEN_DATE)
    reopen_times = [reopen_index, reopen_index + Parameter(REOPEN_PARAM_STR)]
    return linear_interp(reopen_times, np.array([0.0, 1.0]))


def get_spatial_stratification(
    compartments: list, 
    model_pops: pd.DataFrame, 
    tex_doc: StandardTexDoc,
    reopen_func: Function,
) -> Stratification:
    strata = model_pops.columns
    reopen_str = 'wa_reopen_period'.replace('_', '\_')
    description = 'All model compartments previously described were further ' \
        f"stratified into strata to represent Western Australia ({strata[0].upper()}) and `{strata[1]}' " \
        'to represent the remaining major jurisdictions of Australia. ' \
        'This approach was adopted to avoid community transmission occurring in Western Australia ' \
        'prior to the date on which Western Australia re-opened its borders to the rest of the country. ' \
        f'To achieve this effect, transmission in {strata[0].upper()} was initially set to zero, ' \
        f'and subsquently scaled up to being equal to that of the {strata[1]} ' \
        f'jurisdictions of Australia over a period that governed by a calibrated ' \
        f"parameter (`{reopen_str}'). "
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
):
    """
    Args:
        model_pops: Patch and age-specific population
        model: The epidemiological model
        start_props: Starting proportions in the vaccination analysis
        vacc_sens: Whether the vaccination analysis being run
        tex_doc: Documentation object
    """
    start_comp = 'susceptible'
    immunity_strata = model.stratifications['immunity'].strata

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

    model.init_population_with_graphobject(Function(get_init_pop, [Parameter('imm_prop')]))
