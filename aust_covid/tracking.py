from pathlib import Path
from jax import numpy as jnp
import numpy as np
import pandas as pd

from summer2 import CompartmentalModel
from summer2.functions.time import get_linear_interpolation_function
from summer2.functions.derived import get_rolling_reduction
from summer2.parameters import Parameter, DerivedOutput, Function

from aust_covid.utils import convolve_probability, build_gamma_dens_interval_func
from aust_covid.inputs import load_household_impacts_data
from general_utils.tex import StandardTexDoc

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / 'supplement'


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


def track_incidence(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
):
    description = 'Age group and strain-specific and overall incidence of SARS-CoV-2 ' \
        '(including episodes that are never detected) is first tracked. ' \
        'This modelled incident infection quantity is not used explicitly in the calibration process, ' \
        'but tracking this process is necessary for the calculation of several other  ' \
        'model outputs, as described below. ' \
        'The point at which new cases contribute to incidence is taken as ' \
        'the time at which symptoms begin in those infected persons who develop symptoms. ' \
        'To account for the observation that infectiousness is often present for ' \
        'a short period of time prior to the onset of symptoms, ' \
        'we estimate incidence from the transition from the first to second ' \
        'chained sequential infectious compartment.\n'
    tex_doc.add_line(description, 'Outputs', subsection='Notifications')

    age_strata = model.stratifications['agegroup'].strata
    strain_strata = model.stratifications['strain'].strata
    for age in age_strata:
        age_str = f'Xagegroup_{age}'
        for strain in strain_strata:
            strain_str = f'Xstrain_{strain}'
            model.request_output_for_flow(
                f'incidence{age_str}{strain_str}', 
                'inf_transition_0', 
                source_strata={'agegroup': age},
                dest_strata={'strain': strain},
                save_results=False,
            )
        model.request_function_output(
            f'incidence{age_str}',
            sum([DerivedOutput(f'incidence{age_str}Xstrain_{strain}') for strain in strain_strata]),
            save_results=False,
        )
    model.request_function_output(
        f'incidence',
        sum([DerivedOutput(f'incidenceXagegroup_{age}') for age in age_strata]),
    )


def track_notifications(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
    moving_average_window: int,
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
    model.request_function_output('notifications', notif_dist_rel_inc)
    model.request_function_output('notifications_ma', Function(get_rolling_reduction(jnp.mean, moving_average_window), [DerivedOutput('notifications')]))

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
        ratio_fig_name,
        'Outputs', 
        subsection='Notifications',
    )
    
    if show_figs:
        survey_fig.show()
        ratio_fig.show()


def track_deaths(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
    moving_average_window: int,
) -> str:
    description = 'Calculation of the COVID-19-specific deaths follows an analogous ' \
        'approach to that described for notifications, ' \
        'except that there is no assumption of partial observation and ' \
        'age-specific infection fatality rates are used. ' \
        'Specifically, for each age group, we first multiply the age-specific incidence ' \
        'by the infection fatality rate for that group. ' \
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
            death_dist_rel_inc = Function(
                convolve_probability, 
                [DerivedOutput(f'incidence{age_str}{strain_str}'), delay]
            ) * Parameter(f'ifr_{age}') * strain_rel_death * Parameter('ifr_adjuster')
            model.request_function_output(f'deaths{age_str}{strain_str}', death_dist_rel_inc, save_results=False)
        model.request_function_output(
            f'deaths{age_str}',
            sum([DerivedOutput(f'deaths{age_str}Xstrain_{strain}') for strain in strain_strata]),
        )
    model.request_function_output(
        'deaths',
        sum([DerivedOutput(f'deathsXagegroup_{age}') for age in agegroups]),
    )
    model.request_function_output(
        'deaths_ma',
        Function(get_rolling_reduction(jnp.mean, moving_average_window), [DerivedOutput('deaths')])
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

    model.request_function_output('adult_seropos_prop_copy', DerivedOutput('adult_seropos_prop') * 1.0)


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
