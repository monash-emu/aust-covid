from jax import numpy as jnp
import numpy as np
import pandas as pd

from summer2 import CompartmentalModel
from summer2.functions.time import get_linear_interpolation_function
from summer2.functions.derived import get_rolling_reduction
from summer2.parameters import Parameter, DerivedOutput, Function

from inputs.constants import TARGETS_AVERAGE_WINDOW, SUPPLEMENT_PATH
from aust_covid.utils import convolve_probability, build_gamma_dens_interval_func
from aust_covid.inputs import load_household_impacts_data
from emutools.tex import StandardTexDoc
from inputs.constants import INFECTION_PROCESSES


def get_cdr_values(
    param: float, 
    test_ratios: np.array,
) -> pd.Series:
    return 1.0 - np.exp(0.0 - param * test_ratios)


def get_param_to_exp_plateau(
    input_ratio: float, 
    cdr_param_target: float,
) -> float:
    """
    Solve the preceding equation for a value of param, given a ratio input.
    """
    return 0.0 - np.log(1.0 - cdr_param_target) / input_ratio


def track_incidence(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
):
    description = 'Age group and strain-specific, and overall incidence of SARS-CoV-2 ' \
        '(including episodes that are never detected) were tracked. ' \
        'This modelled incident quantity was not used explicitly in the calibration process, ' \
        'but tracking this process was necessary for the calculation of several other  ' \
        'model outputs, as described below. ' \
        'The point at which new cases contribute to incidence is taken as ' \
        'the time of symptom onset in those infected persons who do develop symptoms. ' \
        'To account for the observation that infectiousness is often present for ' \
        'a short period prior to the onset of symptoms, ' \
        'we estimate incidence as the transition from the first to second ' \
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
    model.request_function_output('incidence', sum([DerivedOutput(f'incidenceXagegroup_{age}') for age in age_strata]))


def track_notifications(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
    show_figs: bool=False,
) -> tuple:
    description = 'The extent of community testing following symptomatic infection is likely to have declined ' \
        'over the course of 2022. To understand these trends, we first considered data from the \href' \
        '{https://www.abs.gov.au/statistics/people/people-and-communities/household-impacts-covid-19-survey/latest-release}' \
        '{Australian Bureau of Statistics Household Impacts of COVID-19 surveys}, which were undertaken periodically ' \
        'throughout 2022 with standardised questions at each round ' \
        '(downloded on the 12\\textsuperscript{th} June 2023). ' \
        'These surveys reported on several indicators, ' \
        'including the proportion of households reporting a household member with symptoms of cold, flu or COVID-19, ' \
        'and the proportion of households reporting a household member has had a COVID-19 test. ' \
        'We considered that the ratio of the proportion of households reporting having undertaken COVID-19 tests to the ' \
        'proportion of households with a symptomatic member provided the best available estimate of the decline in ' \
        'testing over this period. ' \
        'We define the case detection rate (CDR) as the proportion of all incident SARS-CoV-2 infection episodes ' \
        '(including asymptomatic and undetected episodes) that are captured through the surveillance data we used in calibration. ' \
        'In calibration, we varied the starting case detection rate at the time of the first survey through plausible ranges, ' \
        'which declined thereafter according to the survey estimates described. ' \
        'The relationship between CDR and our calculated ratio of testing in households to symptomatic persons in households ' \
        'was defined by an exponential function to ensure that the CDR remained in the domain [0, 1], ' \
        'dropping to zero when household testing reached zero and approaching one as household testing approaches very high levels. ' \
        "Specifically, the case detection rate when the ratio is equal to $r$ with starting CDR of $s$ is given by " \
        "$s = 1 - e^{-p \\times r}$. The value of $p$ is calculated to ensure that $s$ is equal to the intended CDR when $r$ is at its starting value. \n"
    tex_doc.add_line(description, 'Outputs', 'Notifications')

    hh_impact = load_household_impacts_data()
    hh_test_ratio = hh_impact['Proportion testing'] / hh_impact['Proportion symptomatic']

    exp_param = get_param_to_exp_plateau(hh_test_ratio[0], Parameter('start_cdr'))
    cdr_values = get_cdr_values(exp_param, hh_test_ratio.to_numpy())

    ratio_interp = get_linear_interpolation_function(jnp.array(model.get_epoch().datetime_to_number(hh_test_ratio.index)), cdr_values)
    tracked_ratio_interp = model.request_track_modelled_value('ratio_interp', ratio_interp)
    
    delay = build_gamma_dens_interval_func(Parameter('notifs_shape'), Parameter('notifs_mean'), model.times)
    model.request_function_output('notifications', Function(convolve_probability, [DerivedOutput('incidence'), delay]) * tracked_ratio_interp)
    model.request_function_output('notifications_ma', Function(get_rolling_reduction(jnp.mean, TARGETS_AVERAGE_WINDOW), [DerivedOutput('notifications')]))

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
) -> str:
    ba2_adj_name = 'ba2_rel_ifr'
    ba2_adj_str = ba2_adj_name.replace('_', '\_')
    description = 'Calculation of the COVID-19-specific deaths follows an analogous ' \
        'approach to that described for notifications, ' \
        'except that there is no assumption of partial observation and ' \
        'age-specific infection fatality rates are used. ' \
        'For each age group, we first multiply the age-specific incidence ' \
        'by the infection fatality rate for that group, ' \
        'and then adjust this rate according to the relative infectiousness of the BA.2 ' \
        f"subvariant in the case of this strain with the `{ba2_adj_str}' parameter. " \
        'Next, we convolve this rate with a gamma distribution ' \
        'to obtain the daily rate of deaths for each age group, and lastly sum over age groups. \n'
    tex_doc.add_line(description, 'Outputs', subsection='Deaths')
    
    agegroups = model.stratifications['agegroup'].strata
    strain_strata = model.stratifications['strain'].strata
    for age in agegroups:
        age_str = f'Xagegroup_{age}'
        for strain in strain_strata:
            strain_str = f'Xstrain_{strain}'
            delay = build_gamma_dens_interval_func(Parameter('deaths_shape'), Parameter('deaths_mean'), model.times)
            death_dist = Function(convolve_probability, [DerivedOutput(f'incidence{age_str}{strain_str}'), delay])
            strain_rel_death = Parameter(ba2_adj_name) if strain == 'ba2' else 1.0
            adjustments = Parameter(f'ifr_{age}') * strain_rel_death * Parameter('ifr_adjuster')
            model.request_function_output(f'deaths{age_str}{strain_str}', death_dist * adjustments, save_results=False)
        age_total = sum([DerivedOutput(f'deaths{age_str}Xstrain_{strain}') for strain in strain_strata])
        model.request_function_output(f'deaths{age_str}', age_total)
    deaths_total = sum([DerivedOutput(f'deathsXagegroup_{age}') for age in agegroups])
    model.request_function_output('deaths', deaths_total)
    deaths_ma = Function(get_rolling_reduction(jnp.mean, TARGETS_AVERAGE_WINDOW), [DerivedOutput('deaths')])
    model.request_function_output('deaths_ma', deaths_ma)


def track_adult_seroprev(
    compartments: list, 
    model: CompartmentalModel,
    adult_cut_off: int,
    tex_doc: StandardTexDoc,
) -> str:
    never_infected_comp = 'susceptible'
    description = 'The proportion of the overall population in any ' \
        f'compartment other than the {never_infected_comp} compartment among those aged {adult_cut_off} years and above ' \
        "is used to estimate the adult `seropositive' proportion. \n"
    tex_doc.add_line(description, 'Outputs', 'Seroprevalence')

    seropos_comps = [comp for comp in compartments if comp != 'susceptible']
    age_strata = model.stratifications['agegroup'].strata
    filter = {'agegroup': age for age in age_strata if int(age) >= adult_cut_off}
    model.request_output_for_compartments(f'adult_pop', compartments, strata=filter, save_results=False)
    model.request_output_for_compartments(f'adult_seropos', seropos_comps, strata=filter, save_results=False)
    model.request_function_output('adult_seropos_prop', DerivedOutput('adult_seropos') / DerivedOutput('adult_pop'))


def track_strain_prop(
    model: CompartmentalModel,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
) -> tuple:
    description = 'Proportional prevalence of each Omicron sub-variant ' \
        'is tracked as the proportion of the population currently in any of ' \
        'the infectious compartments that is infected with the modelled strain of interest ' \
        '(noting that simultaneous infection with multiple strains is not permitted). \n'
    tex_doc.add_line(description, 'Outputs', subsection='Sub-variants')

    model.request_output_for_compartments('prev', infectious_compartments, save_results=False)
    for strain in model.stratifications['strain'].strata:
        model.request_output_for_compartments(f'{strain}_prev', infectious_compartments, {'strain': strain}, save_results=False)
        model.request_function_output(f'{strain}_prop', DerivedOutput(f'{strain}_prev') / DerivedOutput('prev'))


def track_immune_prop(model):
    model_comps = set([c.name for c in model.compartments])
    model.request_output_for_compartments('total_pop', model_comps)
    for stratum in model.stratifications['immunity'].strata:
        model.request_output_for_compartments(f'number_{stratum}', model_comps, {'immunity': stratum})
        model.request_function_output(f'prop_{stratum}', DerivedOutput(f'number_{stratum}') / DerivedOutput('total_pop'))


def track_reproduction_number(
    model: CompartmentalModel,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
):
    description = 'The time-varying effective reproduction number is calculated as ' \
        'the rate of all infections (including both first infection and reinfection) ' \
        'divided by the prevalence of infectious persons (i.e. in the infectious compartments) ' \
        'multiplied by the duration of the infectious period. ' \
        'This quantity was tracked for illustrative purposes, but did not contribute to calibration. \n'
    tex_doc.add_line(description, 'Outputs', subsection='Reproduction Number')

    model.request_output_for_compartments('n_infectious', infectious_compartments)
    for process in INFECTION_PROCESSES:
        model.request_output_for_flow(process, process, save_results=False)
    model.request_function_output('all_infection', sum([DerivedOutput(process) for process in INFECTION_PROCESSES]), save_results=False)
    model.request_function_output('reproduction_number', DerivedOutput('all_infection') / DerivedOutput('n_infectious') * Parameter('infectious_period'))
