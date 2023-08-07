from pathlib import Path
from datetime import datetime
from jax import numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'
from copy import copy

from summer2.functions.time import get_linear_interpolation_function
from summer2 import CompartmentalModel, Stratification, StrainStratification, Multiply
from summer2.parameters import Parameter, DerivedOutput, Function, Time

from aust_covid.model_utils import triangle_wave_func, convolve_probability, build_gamma_dens_interval_func
from aust_covid.inputs import load_pop_data, load_uk_pop_data, load_household_impacts_data
from general_utils.tex_utils import StandardTexDoc

MATRIX_LOCATIONS = [
    'school', 
    'home', 
    'work', 
    'other_locations',
]

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / 'supplement'
DATA_PATH = BASE_PATH / 'data'


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""


def build_model(
    ref_date: datetime, 
    start_date: datetime, 
    end_date: datetime, 
    tex_doc: StandardTexDoc,
):
    
    # Model construction
    n_latent_comps = 1
    n_infectious_comps = n_latent_comps
    latent_compartments = [f'latent_{i}' for i in range(n_latent_comps)]
    infectious_compartments = [f'infectious_{i}' for i in range(n_infectious_comps)]
    compartments = ['susceptible', 'recovered', 'waned'] + infectious_compartments + latent_compartments
    age_strata = list(range(0, 80, 5))
    strain_strata = ['ba1', 'ba2', 'ba5']
    infection_processes = ['infection', 'early_reinfection', 'late_reinfection']

    aust_model = build_base_model(ref_date, compartments, infectious_compartments, start_date, end_date, tex_doc)
    model_pops = load_pop_data(age_strata, tex_doc)
    set_starting_conditions(aust_model, model_pops, tex_doc)
    add_infection(aust_model, latent_compartments, tex_doc)
    add_latent_transition(aust_model, latent_compartments, infectious_compartments, tex_doc)
    add_infectious_transition(aust_model, latent_compartments, infectious_compartments, tex_doc)
    add_waning(aust_model, tex_doc)

    # Age and heterogeneous mixing
    raw_matrices = {i: pd.read_csv(DATA_PATH / f'{i}.csv', index_col=0).to_numpy() for i in MATRIX_LOCATIONS}
    adjusted_matrices = adapt_gb_matrices_to_aust(age_strata, raw_matrices, model_pops, tex_doc)
    mixing_matrix = sum(list(adjusted_matrices.values()))
    age_strat = get_age_stratification(compartments, age_strata, mixing_matrix, tex_doc)
    aust_model.stratify_with(age_strat)

    # Other stratifications and reinfection
    strain_strat = get_strain_stratification(compartments, strain_strata, tex_doc)
    aust_model.stratify_with(strain_strat)
    seed_vocs(aust_model, latent_compartments, tex_doc)

    add_reinfection(aust_model, strain_strata, latent_compartments, tex_doc)

    vacc_strat = get_vacc_stratification(compartments, infection_processes, tex_doc)
    aust_model.stratify_with(vacc_strat)

    spatial_strat = get_spatial_stratification(datetime(2022, 3, 3), compartments, infection_processes, model_pops, aust_model, tex_doc)
    aust_model.stratify_with(spatial_strat)
    adjust_state_pops(model_pops, aust_model, tex_doc)

    # Outputs
    track_incidence(aust_model, infection_processes, tex_doc)
    add_notifications_output(aust_model, tex_doc)
    add_death_output(aust_model, tex_doc)
    track_adult_seroprev(compartments, aust_model, 15, tex_doc)
    track_strain_prop(aust_model, infectious_compartments, tex_doc)
    track_reproduction_number(aust_model, infection_processes, infectious_compartments, tex_doc)

    for comp in compartments:
        aust_model.request_output_for_compartments(comp, [comp])

    return aust_model


def build_base_model(
    ref_date: datetime,
    compartments: list,
    infectious_compartments: list,
    start_date: datetime,
    end_date: datetime,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = f'The base model consists of {len(compartments)} states, ' \
        f'representing the following states: {", ".join(compartments)}. ' \
        f"Only the infectious compartment compartment contributes to the force of infection. " \
        f'The model is run from {start_date.strftime("%d %B %Y")} to {end_date.strftime("%d %B %Y")}. '
    tex_doc.add_line(description, 'Model Construction')

    return CompartmentalModel(
        times=(
            (start_date - ref_date).days, 
            (end_date - ref_date).days,
        ),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        ref_date=ref_date,
    )


def set_starting_conditions(
    model: CompartmentalModel,
    pop_data: pd.DataFrame,
    tex_doc: StandardTexDoc,
) -> str:
    total_pop = pop_data.sum().sum()
    description = f'The simulation starts with {str(round(total_pop / 1e6, 3))} million fully susceptible persons, ' \
        'with infectious persons introduced later through strain seeding as described below. '
    tex_doc.add_line(description, 'Model Construction')

    model.set_initial_population({'susceptible': total_pop})


def add_infection(
    model: CompartmentalModel,
    latent_compartments: list,
    tex_doc: StandardTexDoc,
) -> str:
    process = 'infection'
    origin = 'susceptible'
    destination = latent_compartments[0]
    description = f'The {process} process moves people from the {origin} ' \
        f'compartment to the {destination} compartment, ' \
        'under the frequency-dependent transmission assumption. '
    tex_doc.add_line(description, 'Model Construction')

    model.add_infection_frequency_flow(process, Parameter('contact_rate'), origin, destination)


def add_latent_transition(
    model: CompartmentalModel,
    latent_compartments: list,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
):
    parameter_name = 'latent_period'
    final_dest = infectious_compartments[0]
    n_latent_comps = len(latent_compartments)
    description = 'Following infection, infected persons enter a series of latent compartments. ' \
        f'These are chained in series, with a total of {n_latent_comps} linked together in sequence. ' \
        'To achieve the same '
    tex_doc.add_line(description, 'Model Construction')

    rate = 1.0 / Parameter(parameter_name) * n_latent_comps
    for i_comp in range(n_latent_comps - 1):
        origin = latent_compartments[i_comp]
        destination = latent_compartments[i_comp + 1]
        model.add_transition_flow(f'latent_transition_{str(i_comp)}', rate, origin, destination)
    final_origin = 'susceptible' if n_latent_comps == 1 else destination
    model.add_transition_flow('progression', rate, final_origin, final_dest)


def add_infectious_transition(
    model: CompartmentalModel,
    latent_compartments: list,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
):
    parameter_name = 'infectious_period'
    final_dest = 'recovered'
    n_inf_comps = len(infectious_compartments)
    description = ''  # Need to comment properly
    tex_doc.add_line(description, 'Model Construction')

    rate = 1.0 / Parameter(parameter_name) * n_inf_comps
    for i_comp in range(n_inf_comps - 1):
        origin = infectious_compartments[i_comp]
        destination = infectious_compartments[i_comp + 1]
        model.add_transition_flow(f'inf_transition_{str(i_comp)}', rate, origin, destination)
    final_origin = latent_compartments[-1] if n_inf_comps == 1 else destination
    model.add_transition_flow('recovery', rate, final_origin, final_dest)


def add_waning(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
) -> str:
    process = 'waning'
    origin = 'recovered'
    destination = 'waned'
    parameter_name = 'natural_immunity_period'
    description = 'A waned compartment is included in the model ' \
        'to represent persons who no longer have immunity from past natural immunity. ' \
        f'As these persons lose their infection-induced immunity, they transition from the ' \
        f'{origin} compartment to the {destination} compartment at a rate equal to the reciprocal of the ' \
        f'{parameter_name.replace("_", " ")}. '
    tex_doc.add_line(description, 'Model Construction')

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

    matrix_fig.write_image(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        f'Daily contact rates by age group (row), contact age group (column) and location (panel) for {filename.replace("_", " ").replace(".jpg", "")}. ', 
        filename,
        'Population Mixing', 
    )
    return matrix_fig


def adapt_gb_matrices_to_aust(
    age_strata: list,
    unadjusted_matrices: dict, 
    pop_data: pd.DataFrame,
    tex_doc: StandardTexDoc,
    show_figs: bool=False,
) -> tuple:
    description = 'Matrices were adjusted to account for the differences in the age distribution of the ' \
        'Australian population distribution in 2022 compared to the population of Great Britain in 2000. ' \
        'The matrices were adjusted by taking the dot product of the location-specific unadjusted matrices and the diagonal matrix ' \
        'containing the vector of the ratios between the proportion of the British and Australian populations ' \
        'within each age bracket as its diagonal elements. ' \
        'To align with the methodology of the POLYMOD study \cite{mossong2008} ' \
        'we sourced the 2001 UK census population for those living in the UK at the time of the census ' \
        'from the \href{https://ec.europa.eu/eurostat}{Eurostat database}. '
    tex_doc.add_line(description, 'Population Mixing')

    # Australia
    aust_props_disp = copy(pop_data)
    aust_props_disp['age_group'] = [f'{age}-{age + 4}' for age in age_strata[:-1]] + ['75 and over']
    input_pop_fig = px.bar(
        aust_props_disp.melt(id_vars=['age_group']), 
        x='age_group', 
        y='value', 
        color='variable', 
        labels={'value': 'population', 'age_group': ''},
    )

    input_pop_filename = 'input_population.jpg'
    input_pop_fig.write_image(SUPPLEMENT_PATH / input_pop_filename)
    tex_doc.include_figure(
        'Australian population sizes implemented in the model obtained from Australia Bureau of Statistics.', 
        input_pop_filename,
        'Model Construction', 
    )

    # UK
    raw_uk_data = load_uk_pop_data()
    uk_pop_fig = px.bar(raw_uk_data)
    uk_pop_fig.update_layout(showlegend=False)

    uk_pop_filename = 'uk_population.jpg'
    uk_pop_fig.write_image(SUPPLEMENT_PATH / uk_pop_filename)
    tex_doc.include_figure(
        'United Kingdom population sizes used in matrix weighting.', 
        uk_pop_filename,
        'Population Mixing', 
    )

    # Make weighting calculations
    aust_age_props = pop_data.sum(axis=1) / pop_data.sum().sum()
    uk_age_pops = raw_uk_data[:15]
    uk_age_pops['75 or over'] = raw_uk_data[15:].sum()
    uk_age_pops.index = age_strata
    uk_age_props = uk_age_pops / uk_age_pops.sum()
    aust_uk_ratios = aust_age_props / uk_age_props

    # Check and adjust each location-specific matrix
    adjusted_matrices = {}
    for location in MATRIX_LOCATIONS:
        unadjusted_matrix = unadjusted_matrices[location]
        assert unadjusted_matrix.shape[0] == unadjusted_matrix.shape[1], 'Unadjusted mixing matrix not square'
        assert len(aust_age_props) == unadjusted_matrix.shape[0], 'Different number of Aust age groups from mixing categories'
        assert len(uk_age_props) == unadjusted_matrix.shape[0], 'Different number of UK age groups from mixing categories'
        adjusted_matrices[location] = np.dot(unadjusted_matrix, np.diag(aust_uk_ratios))
    
    # Some final plotting
    raw_matrix_fig = plot_mixing_matrices(unadjusted_matrices, age_strata, 'raw_matrices.jpg', tex_doc)
    adj_matrix_fig = plot_mixing_matrices(adjusted_matrices, age_strata, 'adjusted_matrices.jpg', tex_doc)
    if show_figs:
        input_pop_fig.show()
        uk_pop_fig.show()
        raw_matrix_fig.show()
        adj_matrix_fig.show()

    return adjusted_matrices


def get_age_stratification(
    compartments: list,
    age_strata: list,
    matrix: np.array,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = 'We stratified all compartments of the base model ' \
        'into sequential age brackets in five year ' \
        f'bands from age {age_strata[0]} to {age_strata[0] + 4} through to age {age_strata[-2]} to {age_strata[-2] + 4} ' \
        f'with a final age band to represent those aged {age_strata[-1]} and above. ' \
        'These age brackets were chosen to match those used by the POLYMOD survey and so fit with the mixing data available. ' \
        'The population distribution by age group was informed by the data from the Australian ' \
        'Bureau of Statistics introduced previously. '
    tex_doc.add_line(description, 'Stratification', subsection='Age')

    age_strat = Stratification('agegroup', age_strata, compartments)
    age_strat.set_mixing_matrix(matrix)
    return age_strat


def get_strain_stratification(
    compartments: list,
    strain_strata,
    tex_doc: StandardTexDoc,
) -> tuple:
    strain_strings = [f'{strain.replace("ba", "BA.")}' for strain in strain_strata]
    compartments_to_stratify = [comp for comp in compartments if comp != 'susceptible']
    description = f'We stratified the following compartments according to strain: {", ".join(compartments_to_stratify)}, ' \
        f'including compartments to represent strains: {", ".join(strain_strings)}. ' \
        f"This was implemented using summer's `{StrainStratification.__name__}' class. "
    tex_doc.add_line(description, 'Stratification', subsection='Omicron Sub-variants')

    return StrainStratification('strain', strain_strata, compartments_to_stratify)


def seed_vocs(
    model: CompartmentalModel,
    latent_compartments: list,
    tex_doc: StandardTexDoc,
) -> str:
    strains = model.stratifications['strain'].strata
    seed_comp = latent_compartments[0]
    seed_duration_str = 'seed_duration'
    seed_rate_str = 'seed_rate'
    description = f'Each strain (including the starting {strains[0].replace("ba", "BA.")} strain) is seeded through ' \
        'a step function that allows the introduction of a constant rate of new infectious ' \
        f'persons into the {seed_comp} compartment over a fixed seeding duration defined by a single ' \
        f'{seed_duration_str.replace("_", " ")} parameter. ' \
        f'and at a rate defined by one {seed_rate_str.replace("_", " ")} parameter. ' \
        'The time of first emergence of each strain into the system is defined by ' \
        'a separate emergence time parameter for each strain. '
    tex_doc.add_line(description, 'Stratification', subsection='Omicron Sub-variants')

    for strain in strains:
        voc_seed_func = Function(
            triangle_wave_func, 
            [
                Time,
                Parameter(f'{strain}_seed_time'), 
                Parameter(seed_duration_str), 
                Parameter(seed_rate_str),
            ]
        )
        model.add_importation_flow(
            f'seed_{strain}',
            voc_seed_func,
            seed_comp,
            dest_strata={'strain': strain},
            split_imports=True,
        )


def add_reinfection(
    model: CompartmentalModel,
    strain_strata: list,
    latent_compartments: list,
    tex_doc: StandardTexDoc,
) -> str:
    destination = latent_compartments[0]
    description = 'Reinfection is possible from both the recovered ' \
        'and waned compartments, with these processes termed ' \
        "`early' and `late' reinfection respectively. " \
        'In the case of early reinfection, this is only possible ' \
        'for persons who have recovered from an earlier circulating sub-variant. ' \
        'That is, BA.2 early reinfection is possible for persons previously infected with ' \
        'BA.1, while BA.5 reinfection is possible for persons previously infected with ' \
        'BA.1 or BA.2. The degree of immune escape is determined by the infecting variant ' \
        'and differs for BA.2 and BA.5. This implies that the rate of reinfection ' \
        'is equal for BA.5 reinfecting those recovered from past BA.1 infection ' \
        'as it is for those recovered from past BA.2 infection. ' \
        'For late reinfection, all natural immunity is lost for persons in the waned compartment, ' \
        'such that the rate of reinfection for these persons is the same as the rate of infection ' \
        'for fully susceptible persons. ' \
        'As for the first infection process, all reinfection processes transition individuals ' \
        'to the latent compartment corresponding to the infecting strain.\n'
    tex_doc.add_line(description, 'Reinfection')

    for dest_strain in strain_strata:
        for source_strain in strain_strata:
            process = 'early_reinfection'
            origin = 'recovered'
            escape = Parameter(f'{dest_strain}_escape') if int(dest_strain[-1]) > int(source_strain[-1]) else 0.0
            model.add_infection_frequency_flow(
                process, 
                Parameter('contact_rate') * escape,
                origin, 
                destination,
                source_strata={'strain': source_strain},
                dest_strata={'strain': dest_strain},
            )
            process = 'late_reinfection'
            origin = 'waned'
            model.add_infection_frequency_flow(
                process, 
                Parameter('contact_rate'),
                origin, 
                destination,
                source_strata={'strain': source_strain},
                dest_strata={'strain': dest_strain},
            )


def get_vacc_stratification(
    compartments: list, 
    infection_processes: list,
    tex_doc: StandardTexDoc,
) -> Stratification:
    description = 'All compartments and stratifications described are further ' \
        'stratified into two strata with differing levels of susceptibility to infection. ' \
        'These are loosely intended to represent persons with and without protection ' \
        'against infection attributable to vaccination. ' \
        'One parameter is used to represent the proportion of the population retaining ' \
        'immunological protection against infection through vaccination, ' \
        'with a second parameter used to quantify the relative reduction in ' \
        'the rate of infection and reinfection for those in the stratum with ' \
        'reduced susceptibility.\n'
    tex_doc.add_line(description, 'Stratification', subsection='Vaccination')

    vacc_strat = Stratification('vaccination', ['vacc', 'unvacc'], compartments)
    for infection_process in infection_processes:
        vacc_strat.set_flow_adjustments(
            infection_process,
            {
                'vacc': Multiply(1.0 - Parameter('vacc_infect_protect')),
                'unvacc': None,
            },
        )
    vacc_strat.set_population_split(
        {
            'vacc': Parameter('vacc_prop'),
            'unvacc': 1.0 - Parameter('vacc_prop'),
        }
    )
    return vacc_strat


def get_spatial_stratification(
    reopen_date: datetime, 
    compartments: list, 
    infection_processes: list, 
    model_pops: pd.DataFrame, 
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
) -> Stratification:
    strata = model_pops.columns
    wa_reopen_period = 30.0
    description = 'All model compartments previously described are further ' \
        f"stratified into strata to represent {strata[0].upper()} and `{strata[1]}' " \
        'to represent the remaining major jurisdictions of Australia. ' \
        f'Transmission in {strata[0].upper()} is initially set to zero, ' \
        f'and subsquently scaled up to being equal to the {strata[1]} ' \
        f'jurisdictions of Australia over an arbitrary period of {wa_reopen_period} days. '
    tex_doc.add_line(description, 'Stratification', subsection='Spatial')

    spatial_strat = Stratification('states', model_pops.columns, compartments)
    state_props = model_pops.sum() / model_pops.sum().sum()
    spatial_strat.set_population_split(state_props.to_dict())

    reopen_index = model.get_epoch().dti_to_index(reopen_date)
    reopen_func = get_linear_interpolation_function(
        [reopen_index, reopen_index + wa_reopen_period],
        np.array([0.0, 1.0]),
    )
    infection_adj = {strata[0]: reopen_func, strata[1]: None}
    for infection_process in infection_processes:
        spatial_strat.set_flow_adjustments(infection_process, infection_adj)

    return spatial_strat


def adjust_state_pops(
    model_pops: pd.DataFrame, 
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
):
    strata = model_pops.columns
    description = 'Starting model populations are distributed by ' \
        f'age and spatial status ({strata[0].upper()}, {strata[1]}) ' \
        'according to the age distribution in each simulated region. '
    tex_doc.add_line(description, 'Stratification', 'Spatial')

    for state in model_pops.columns:
        props = model_pops[state] / model_pops[state].sum()
        props.index = props.index.astype(str)
        model.adjust_population_split('agegroup', {'states': state}, props.to_dict())


def track_incidence(
    model: CompartmentalModel,
    infection_processes: list,
    tex_doc: StandardTexDoc,
):
    description = 'Age group and strain-specific and overall incidence of infection with SARS-CoV-2 ' \
        '(including episodes that are never detected) is first tracked. ' \
        'This modelled incident infection quantity is not used explicitly in the calibration process, ' \
        'but tracking this process is necessary for the calculation of several other  ' \
        'model outputs, as described below.\n'
    tex_doc.add_line(description, 'Outputs', subsection='Notifications')

    age_strata = model.stratifications['agegroup'].strata
    strain_strata = model.stratifications['strain'].strata
    for age in age_strata:
        age_str = f'Xagegroup_{age}'
        for strain in strain_strata:
            strain_str = f'Xstrain_{strain}'
            for process in infection_processes:
                model.request_output_for_flow(
                    f'{process}_onset{age_str}{strain_str}', 
                    process, 
                    source_strata={'agegroup': age},
                    dest_strata={'strain': strain},
                    save_results=False,
                )
            model.request_function_output(
                f'incidence{age_str}{strain_str}',
                func=sum([DerivedOutput(f'{process}_onset{age_str}{strain_str}') for process in infection_processes]),
                save_results=False,
            )
        model.request_function_output(
            f'incidence{age_str}',
            func=sum([DerivedOutput(f'incidence{age_str}Xstrain_{strain}') for strain in strain_strata]),
            save_results=False,
        )
    model.request_function_output(
        f'incidence',
        func=sum([DerivedOutput(f'incidenceXagegroup_{age}') for age in age_strata]),
    )


def get_cdr_values(
    param: float, 
    test_ratios: np.array,
) -> pd.Series:
    return 1.0 - np.exp(0.0 - param * test_ratios)


def get_param_to_exp_plateau(
    input_request: float, 
    output_request: float,
) -> float:
    return 0.0 - np.log(1.0 - output_request) / input_request


def add_notifications_output(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
    show_figs: bool=False,
) -> tuple:
    description = 'The extent of community testing following symptomatic infection is likely to have declined ' \
        'over the course of 2022. To understand these trends, we first considered data from the \href' \
        '{https://www.abs.gov.au/statistics/people/people-and-communities/household-impacts-covid-19-survey/latest-release}' \
        '{Australian Bureau of Statistics Household Impacts of COVID-19 surveys} which were undertaken periodically ' \
        'throughout 2022 with standardised questions at each round ' \
        '(downloded on the 12\\textsuperscript{th} June 2023). ' \
        'These surveys reported on several indicators, ' \
        'including the proportion of households reporting a household member with symptoms of cold, flu or COVID-19, ' \
        'and the proportion of households reporting a household member has had a COVID-19 test. ' \
        'We considered that the ratio of the proportion of households reporting having undertaken COVID-19 tests to the ' \
        'proportion of households with a symptomatic member provided the best available estimate of the decline in ' \
        'testing over this period. ' \
        'We define the case detection rate (CDR) as the proportion of all incident COVID-19 episodes ' \
        '(including asymptomatic and undetected episodes) that are captured through the surveillance data we used in calibration. ' \
        'In calibration, we varied the starting case detection rate at the time of the first survey through plausible ranges, ' \
        'which declined thereafter according to the survey estimates descrbied. ' \
        'The relationship between CDR and our calculated ratio of testing in households to symptomatic persons in households ' \
        'was defined by an exponential function to ensure that the CDR remained in the domain [0, 1], ' \
        'dropping to zero when household testing reached zero and approached one as household testing approached very high levels. ' \
        "Specifically, the case detection rate when the ratio is equal to $r$ with starting CDR of $s$ is given by " \
        "$s = (1 - e^{-p \\times r})$. The value of $p$ is calculated to ensure that $s$ is equal to the intended CDR when $r$ is at its starting value.\n"
    tex_doc.add_line(description, 'Outputs', 'Notifications')

    hh_impact = load_household_impacts_data()
    hh_test_ratio = hh_impact['Proportion testing'] / hh_impact['Proportion symptomatic']

    exp_param = get_param_to_exp_plateau(hh_test_ratio[0], Parameter('start_cdr'))
    cdr_values = get_cdr_values(exp_param, hh_test_ratio.to_numpy())

    aust_epoch = model.get_epoch()
    ratio_interp = get_linear_interpolation_function(jnp.array(aust_epoch.datetime_to_number(hh_test_ratio.index)), cdr_values)
    tracked_ratio_interp = model.request_track_modelled_value("ratio_interp", ratio_interp)
    
    delay = build_gamma_dens_interval_func(Parameter('notifs_shape'), Parameter('notifs_mean'), model.times)

    notif_dist_rel_inc = Function(convolve_probability, [DerivedOutput('incidence'), delay]) * tracked_ratio_interp
    model.request_function_output(name='notifications', func=notif_dist_rel_inc)

    survey_fig = hh_impact.plot(labels={'value': 'percentage', 'index': ''}, markers=True)
    
    survey_fig_name = 'survey.jpg'
    survey_fig.write_image(SUPPLEMENT_PATH / survey_fig_name)
    tex_doc.include_figure(
        'Raw survey values from Household Impacts of COVID-19 surveys. ', 
        survey_fig_name,
        'Outputs', 
        subsection='Notifications',
    )
    
    ratio_fig = hh_test_ratio.plot(labels={'value': 'ratio', 'index': ''}, markers=True)
    ratio_fig.update_layout(showlegend=False)
    
    ratio_fig_name = 'ratio.jpg'
    ratio_fig.write_image(SUPPLEMENT_PATH / ratio_fig_name)
    tex_doc.include_figure(
        'Ratio of proportion of households testing to proportion reporting symptoms.', 
        survey_fig_name,
        'Outputs', 
        subsection='Notifications',
    )
    
    if show_figs:
        survey_fig.show()
        ratio_fig.show()


def add_death_output(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
) -> str:
    description = 'Calculation of the COVID-19-specific deaths follows an analogous ' \
        'approach to that described for notifications, ' \
        'except that there is no assumption of partial observation and ' \
        'age-specific infection fatality rates are used. ' \
        'Specifically, for each age group, we first multiply the age-specific incidence ' \
        'by the infection fataliry rate for that group. ' \
        'Next, we convolve this rate with a gamma distribution ' \
        'to obtain the daily rate of deaths for each age group, and lastly sum over age groups.\n'
    tex_doc.add_line(description, 'Outputs', subsection='Deaths')
    
    agegroups = model.stratifications['agegroup'].strata
    strain_strata = model.stratifications['strain'].strata
    for age in agegroups:
        age_str = f'Xagegroup_{age}'
        for strain in strain_strata:
            strain_rel_death = Parameter('ba2_rel_ifr') if strain == 'ba2' else 1.0
            strain_str = f'Xstrain_{strain}'
            delay = build_gamma_dens_interval_func(Parameter('deaths_shape'), Parameter('deaths_mean'), model.times)
            death_dist_rel_inc = Function(convolve_probability, [DerivedOutput(f'incidence{age_str}{strain_str}'), delay]) * Parameter(f'ifr_{age}') * strain_rel_death
            model.request_function_output(name=f'deaths{age_str}{strain_str}', func=death_dist_rel_inc, save_results=False)
        model.request_function_output(
            f'deaths{age_str}',
            func=sum([DerivedOutput(f'deaths{age_str}Xstrain_{strain}') for strain in strain_strata]),
        )
    model.request_function_output(
        'deaths',
        func=sum([DerivedOutput(f'deathsXagegroup_{age}') for age in agegroups]),
    )


def track_adult_seroprev(
    compartments: list, 
    model: CompartmentalModel,
    adult_cut_off: int,
    tex_doc: StandardTexDoc,
) -> str:
    never_infected_comp = 'susceptible'
    description = 'The proportion of the overall population in any ' \
        f'compartment other than the {never_infected_comp} in those aged 15 years and above ' \
        "is used to estimate the adult `seropositive' proportion.\n"
    tex_doc.add_line(description, 'Outputs', 'Seroprevalence')

    seropos_comps = [comp for comp in compartments if comp != 'susceptible']
    age_strata = model.stratifications['agegroup'].strata
    filters = {
        'child': {'agegroup': age for age in age_strata if int(age) < adult_cut_off},
        'adult': {'agegroup': age for age in age_strata if int(age) >= adult_cut_off},
    }
    for age_cat in ['child', 'adult']:
        model.request_output_for_compartments(f'{age_cat}_pop', compartments, strata=filters[age_cat], save_results=False)
        model.request_output_for_compartments(f'{age_cat}_seropos', seropos_comps, strata=filters[age_cat], save_results=False)
        model.request_function_output(f'{age_cat}_seropos_prop', DerivedOutput(f'{age_cat}_seropos') / DerivedOutput(f'{age_cat}_pop'))


def track_strain_prop(
    model: CompartmentalModel,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = 'Proportional prevalence of each Omicron sub-variant ' \
        'is tracked as the proportion of the population currently in ' \
        'the infectious compartment that is infected with the modelled strain of interest ' \
        '(noting that simultaneous infection with multiple strains is not modelled).\n'
    tex_doc.add_line(description, 'Outputs', subsection='Sub-variants')

    model.request_output_for_compartments('prev', infectious_compartments, save_results=False)
    for strain in model.stratifications['strain'].strata:
        model.request_output_for_compartments(f'{strain}_prev', infectious_compartments, {'strain': strain}, save_results=False)
        model.request_function_output(f'{strain}_prop', DerivedOutput(f'{strain}_prev') / DerivedOutput('prev'))


def track_reproduction_number(
    model: CompartmentalModel,
    infection_processes: list,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
):
    description = 'The time-varying effective reproduction number is calculated as ' \
        'the rate of all infections (including both first infection and reinfection) ' \
        'divided by the prevalence of infectious persons (i.e. in the infectious compartments) ' \
        'multiplied by the duration of the infectious period.\n'
    tex_doc.add_line(description, 'Outputs', subsection='Reproduction Number')

    model.request_output_for_compartments('n_infectious', infectious_compartments)
    for process in infection_processes:
        model.request_output_for_flow(process, process, save_results=False)
    model.request_function_output('all_infection', sum([DerivedOutput(process) for process in infection_processes]), save_results=False)
    model.request_function_output('reproduction_number', DerivedOutput('all_infection') / DerivedOutput('n_infectious') * Parameter('infectious_period'))


# Have left these last two functions for now because they are or should be more related to
# the calibration process.

# def show_cdr_profiles(
#     start_cdr_samples: pd.Series, 
#     hh_test_ratio: pd.Series,
# ) -> tuple:
#     cdr_values = pd.DataFrame()
#     for start_cdr in start_cdr_samples:
#         exp_param = get_param_to_exp_plateau(hh_test_ratio[0], start_cdr)
#         cdr_values[round(start_cdr, 3)] = get_cdr_values(exp_param, hh_test_ratio)

#     modelled_cdr_fig_name = "modelled_cdr.jpg"
#     modelled_cdr_fig = cdr_values.plot(markers=True, labels={"value": "case detection ratio", "index": ""})
#     modelled_cdr_fig.write_image(SUPPLEMENT_PATH / modelled_cdr_fig_name)
#     modelled_cdr_fig_caption = "Example case detection rates implemented in randomly selected model runs."

#     return modelled_cdr_fig, modelled_cdr_fig_name, modelled_cdr_fig_caption
#
#
# def show_strain_props(
#     strain_strata: list, 
#     plot_start_time: datetime.date,
#     model: CompartmentalModel,
# ) -> tuple:
#     """
#     Args:
#         strain_strata: Names of sub-variants
#         plot_start_time: Request for left-hand end point for x-axis
#         model: Working model

#     Returns:
#         Output figure, name used to save figure, caption for figure
#     """
    
#     end_date = model.get_epoch().index_to_dti([model.times[-1]])  # Plot to end of simulation
#     strain_props = [f"{strain}_prop" for strain in strain_strata]
#     strain_prop_fig_name = "strain_prop.jpg"
#     strain_prop_fig = model.get_derived_outputs_df()[strain_props].plot.area(labels={"value": "proportion", "index": ""})
#     voc_emerge_df = pd.DataFrame(
#         {
#             "ba1": [datetime(2021, 11, 22), datetime(2021, 11, 29), datetime(2021, 12, 20), "blue"],
#             "ba2": [datetime(2021, 11, 29), datetime(2022, 1, 10), datetime(2022, 3, 7), "red"], 
#             "ba5": [datetime(2022, 3, 28), datetime(2022, 5, 16), datetime(2022, 6, 27), "green"],
#         },
#         index=["any", ">1%", ">50%", "colour"]
#     )
#     lag = timedelta(days=3.5)  # Dates are given as first day of week in which VoC was first detected
#     for voc in voc_emerge_df:
#         voc_info = voc_emerge_df[voc]
#         colour = voc_info["colour"]
#         strain_prop_fig.add_vline(voc_info["any"] + lag, line_dash="dot", line_color=colour)
#         strain_prop_fig.add_vline(voc_info[">1%"] + lag, line_dash="dash", line_color=colour)
#         strain_prop_fig.add_vline(voc_info[">50%"] + lag, line_color=colour)
#     strain_prop_fig.update_xaxes(range=(plot_start_time, end_date[0]))
#     strain_prop_fig.update_yaxes(range=(0.0, 1.0))
#     strain_prop_fig.write_image(SUPPLEMENT_PATH / strain_prop_fig_name)
#     strain_prop_fig_caption = "Proportion of prevalent cases by sub-variant, with first sequence proportion times. " \
#         "Dotted line, first isolate of VoC; dashed line, first time VoC represents more than 1% of all isolates; " \
#         "solid line, first time VoC represnets more than 50% of all isolates. "
#     return strain_prop_fig, strain_prop_fig_name, strain_prop_fig_caption
