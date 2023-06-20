from datetime import datetime, timedelta
from jax import numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

from summer2.functions.time import get_linear_interpolation_function
from summer2 import CompartmentalModel, Stratification, StrainStratification
from summer2.parameters import Parameter, DerivedOutput, Function, Time

from aust_covid.model_utils import triangle_wave_func, convolve_probability, build_gamma_dens_interval_func
from aust_covid.inputs import load_pop_data, load_uk_pop_data, load_household_impacts_data, load_google_mob_year_df

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""


def get_param_to_exp_plateau(
    input_request: float, 
    output_request: float,
) -> float:
    """
    Get the parameter needed to ensure the function:
    output = 1 - exp(-param * input)
    passes through the requested input and output.
    
    Args:
        input_request: Independent variable at known point
        output_request: Dependent variable at known point
    """
    return 0.0 - np.log(1.0 - output_request) / input_request


def build_base_model(
    ref_date: datetime,
    compartments: list,
    start_date: datetime,
    end_date: datetime,
) -> tuple:
    """
    Args:
        ref_date: Arbitrary reference date
        compartments: Starting unstratified compartments
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        Simple model starting point for extension through the following functions
        with text description of the process.
    """
    infectious_compartment = "infectious"
    model = CompartmentalModel(
        times=(
            (start_date - ref_date).days, 
            (end_date - ref_date).days,
        ),
        compartments=compartments,
        infectious_compartments=[infectious_compartment],
        ref_date=ref_date,
    )
    description = f"The base model consists of {len(compartments)} states, " \
        f"representing the following states: {', '.join(compartments)}. " \
        f"Only the {infectious_compartment} compartment contributes to the force of infection. " \
        f"The model is run from {str(start_date.date())} to {str(end_date.date())}. "
    return model, description


def get_pop_data() -> tuple:
    """
    Returns:
        Dataframe containing all Australian population data we may need
        with description.
    """
    pop_data, sheet_name = load_pop_data()
    sheet_name = sheet_name.replace("_", "\\textunderscore")
    description = f"For estimates of the Australian population, the {sheet_name} spreadsheet was downloaded " \
        "from the Australian Bureau of Statistics website on the 1st of March 2023 \cite{abs2022}. "
    return pop_data, description


def set_starting_conditions(
    model: CompartmentalModel,
    pop_data: pd.DataFrame,
    adjuster: 1.0,
) -> str:
    """
    Args:
        model: Working compartmental model
        pop_data: Data on the Australian population

    Returns:
        Description of data being used
    """
    total_pop = pop_data["Australia"].sum() * adjuster
    model.set_initial_population({"susceptible": total_pop})
    return f"The simulation starts with {str(round(total_pop / 1e6, 3))} million fully susceptible persons, " \
        "with infectious persons introduced later through strain seeding as described below. "


def add_infection(
    model: CompartmentalModel,
) -> str:
    """
    Args:
        model: Working compartmental model

    Returns:
        Description of process added
    """
    process = "infection"
    origin = "susceptible"
    destination = "latent"
    model.add_infection_frequency_flow(process, Parameter("contact_rate"), origin, destination)
    return f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "


def add_progression(
    model: CompartmentalModel,
) -> str:
    process = "progression"
    origin = "latent"
    destination = "infectious"
    parameter_name = "latent_period"
    model.add_transition_flow(process, 1.0 / Parameter(parameter_name), origin, destination)
    return f"The {process} process moves " \
        f"people from the {origin} state to the {destination} compartment, " \
        f"with the transition rate calculated as the reciprocal of the {parameter_name.replace('_', ' ')}. "


def add_recovery(
    model: CompartmentalModel,
) -> str:
    process = "recovery"
    origin = "infectious"
    destination = "recovered"
    parameter_name = "infectious_period"
    model.add_transition_flow(process, 1.0 / Parameter(parameter_name), origin, destination)
    return f"The {process} process moves " \
        f"people from the {origin} state to the {destination} compartment, " \
        f"with the transition rate calculated as the reciprocal of the {parameter_name.replace('_', ' ')}. "


def add_waning(
    model: CompartmentalModel,
) -> str:
    process = "waning"
    origin = "recovered"
    destination = "waned"
    parameter_name = "natural_immunity_period"
    model.add_transition_flow(process, 1.0 / Parameter(parameter_name), origin, destination)
    return "A waned compartment is included in the model " \
        "to represent persons who no longer have immunity from past natural immunity. " \
        f"As these persons lose their infection-induced immunity, they transition from the " \
        f"{origin} compartment to the {destination} compartment at a rate equal to the reciprocal of the " \
        f"{parameter_name.replace('_', ' ')}. "


def adapt_gb_matrices_to_aust(
    age_strata: list,
    unadjusted_matrix: np.array, 
    pop_data: pd.DataFrame,
) -> tuple:

    # Australian population distribution by age        
    aust_pop_series = pop_data["Australia"]
    modelled_pops = aust_pop_series[:"70-74"]
    modelled_pops["75"] = aust_pop_series["75-79":].sum()
    modelled_pops.index = age_strata
    aust_age_props = pd.Series([pop / aust_pop_series.sum() for pop in modelled_pops], index=age_strata)
    assert len(aust_age_props) == unadjusted_matrix.shape[0], "Different number of Aust age groups from mixing categories"

    # UK population distributions
    raw_uk_data = load_uk_pop_data()
    uk_age_pops = raw_uk_data[:15]
    uk_age_pops["75 years and up"] = raw_uk_data[15:].sum()
    uk_age_pops.index = age_strata
    uk_age_props = uk_age_pops / uk_age_pops.sum()
    assert len(uk_age_props) == unadjusted_matrix.shape[0], "Different number of UK age groups from mixing categories"

    # Calculation
    aust_uk_ratios = aust_age_props / uk_age_props
    adjusted_matrix = np.dot(unadjusted_matrix, np.diag(aust_uk_ratios))

    aust_age_props.index = aust_age_props.index.astype(str)
    return adjusted_matrix, aust_age_props


def adapt_gb_matrix_to_aust(
    age_strata: list,
    unadjusted_matrix: np.array, 
    pop_data: pd.DataFrame,
) -> tuple:
    """
    Args:
        unadjusted_matrix: The unadjusted matrix
        pop_data: ABS population numbers
    Returns:
        Matrix adjusted to target population
        Proportions of Australian population in modelled age groups
    """
    
    assert unadjusted_matrix.shape[0] == unadjusted_matrix.shape[1], "Unadjusted mixing matrix not square"

    # Australian population distribution by age        
    aust_pop_series = pop_data["Australia"]
    modelled_pops = aust_pop_series[:"65-69"]
    modelled_pops["70"] = aust_pop_series["70-74":].sum()
    modelled_pops.index = age_strata
    aust_age_props = pd.Series([pop / aust_pop_series.sum() for pop in modelled_pops], index=age_strata)
    assert len(aust_age_props) == unadjusted_matrix.shape[0], "Different number of Aust age groups from mixing categories"

    # UK population distributions
    raw_uk_data = load_uk_pop_data()
    uk_age_pops = raw_uk_data[:14]
    uk_age_pops["70 years and up"] = raw_uk_data[14:].sum()
    uk_age_pops.index = age_strata
    uk_age_props = uk_age_pops / uk_age_pops.sum()
    assert len(uk_age_props) == unadjusted_matrix.shape[0], "Different number of UK age groups from mixing categories"
    
    # Calculation
    aust_uk_ratios = aust_age_props / uk_age_props
    adjusted_matrix = np.dot(unadjusted_matrix, np.diag(aust_uk_ratios))
    
    description = "Matrices were adjusted to account for the differences in the age distribution of the " \
        "Australian population distribution in 2022 compared to the population of Great Britain in 2000. " \
        "The matrices were adjusted by taking the dot product of the unadjusted matrices and the diagonal matrix " \
        "containing the vector of the ratios between the proportion of the British and Australian populations " \
        "within each age bracket as its diagonal elements. " \
        "To align with the methodology of the POLYMOD study \cite{mossong2008} " \
        "we sourced the 2001 UK census population for those living in the UK at the time of the census " \
        "from the Eurostat database (https://ec.europa.eu/eurostat). "

    age_group_names = [f"{age}-{age + 4}" for age in age_strata[:-1]] + ["70 and over"]

    input_pop_filename = "input_population.jpg"
    input_pop_fig = px.bar(aust_pop_series, labels={"value": "population", "Age (years)": ""})
    input_pop_fig.update_layout(showlegend=False)
    input_pop_fig.write_image(SUPPLEMENT_PATH / input_pop_filename)
    input_pop_caption = "Australian population sizes by age group obtained from Australia Bureau of Statistics."

    modelled_pop_filename = "modelled_population.jpg"
    modelled_pop_fig = px.bar(modelled_pops, labels={"value": "population", "index": ""})
    modelled_pop_fig.update_layout(xaxis=dict(tickvals=age_strata, ticktext=age_group_names, tickangle=45), showlegend=False)
    modelled_pop_fig.write_image(SUPPLEMENT_PATH / modelled_pop_filename)
    modelled_pop_caption = "Population sizes by age group implemented in the model."

    matrix_ref_pop_filename = "matrix_ref_pop.jpg"
    matrix_ref_pop_fig = px.bar(uk_age_pops, labels={"value": "population", "index": ""})
    matrix_ref_pop_fig.update_layout(xaxis=dict(tickvals=age_strata, ticktext=age_group_names, tickangle=45), showlegend=False)
    matrix_ref_pop_fig.write_image(SUPPLEMENT_PATH / matrix_ref_pop_filename)
    matrix_ref_pop_caption = "United Kingdom population sizes."

    adjusted_matrix_filename = "adjusted_matrix.jpg"
    adjusted_matrix_fig = px.imshow(unadjusted_matrix, x=age_strata, y=age_strata)
    adjusted_matrix_fig.write_image(SUPPLEMENT_PATH / adjusted_matrix_filename)
    adjusted_matrix_caption = "Matrices adjusted to Australian population. Values are contacts per person per day. "

    aust_age_props.index = aust_age_props.index.astype(str)
    return adjusted_matrix, aust_age_props, description, \
        input_pop_filename, input_pop_caption, input_pop_fig, modelled_pop_filename, modelled_pop_caption, modelled_pop_fig, \
        matrix_ref_pop_filename, matrix_ref_pop_caption, matrix_ref_pop_fig, adjusted_matrix_filename, adjusted_matrix_caption, adjusted_matrix_fig, modelled_pops


def get_raw_mobility(
    plot_start_time: datetime.date,
    model: CompartmentalModel,
) -> tuple:
    """
    Args:
        plot_start_time: Left limit of x-axis
        model: The compartmental model

    Returns:
        Raw mobility data
    """
    mob_df = pd.concat([load_google_mob_year_df(2021), load_google_mob_year_df(2022)])
    mob_df.columns = [col.replace("_percent_change_from_baseline", "").replace("_", " ") for col in mob_df.columns]
    end_date = model.get_epoch().index_to_dti([model.times[-1]])
    
    raw_mob_filename = "raw_mobility.jpg"
    raw_mob_fig = mob_df.plot(labels={"value": "percent change from baseline", "date": ""})
    raw_mob_fig.update_xaxes(range=(plot_start_time, end_date[0]))
    raw_mob_fig.write_image(SUPPLEMENT_PATH / raw_mob_filename)

    raw_mob_text = "Google mobility data were downloaded from https://www.gstatic.com/covid19/ mobility/Region_Mobility_Report_CSVs.zip" \
        "on the 15th June 2023. These spreadsheets provide daily estimates of rates of attendance at certain `locations' and can " \
        "be used to provide an overall picture of the populations's attendance at specific venue types. "

    return mob_df, raw_mob_fig, raw_mob_filename, raw_mob_text


def process_mobility(
    mob_df: pd.DataFrame,
    plot_start_time: datetime.date,
    model: CompartmentalModel,
) -> tuple:
    """
    Args:
        mob_df: Raw mobility data returned by previous function
        plot_start_time: Left limit of x-axis
        model: The compartmental model

    Returns:
        Processed mobility data to be used in model
    """
    non_resi_mob = mob_df[[col for col in mob_df.columns if "residential" not in col]]
    mean_mob = non_resi_mob.mean(axis=1)
    smoothed_mean_mob = mean_mob.rolling(window=14).mean().dropna()
    combined_mob = pd.DataFrame({"mean non-residential": mean_mob, "smoothed mean non-resi": smoothed_mean_mob})
    end_date = model.get_epoch().index_to_dti([model.times[-1]])

    modelled_mob_filename = "modelled_mobility.jpg"
    modelled_mob_fig = combined_mob.plot(labels={"value": "percent change from baseline", "date": ""})
    modelled_mob_fig.update_xaxes(range=(plot_start_time, end_date[0]))
    modelled_mob_fig.write_image(SUPPLEMENT_PATH / modelled_mob_filename)
    return smoothed_mean_mob, modelled_mob_fig, modelled_mob_filename


def calculate_mobility_effect(
    mob_input: pd.Series,
    plot_start_time: datetime.date,
    model: CompartmentalModel,
) -> tuple:
    mobility_effect = (1.0 + mob_input / 100.0) ** 2.0
    mobility_effect_func = get_linear_interpolation_function(model.get_epoch().dti_to_index(mobility_effect.index), mobility_effect.to_numpy())

    mob_adj_text = "The adjustment in the rates of contact at the locations affected by mobility is " \
        "calculated as one plus the averaged Google mobility percentage change metric divided by 100 (usually negative). " \
        "This is then squared to account for this effect applying to both the infector and infectee of any potential " \
        "effective contact. "
    mob_effect_filename = "mobility_effect.jpg"
    mob_effect_fig = mobility_effect.plot(labels={"value": "adjustment", "date": ""})
    end_date = model.get_epoch().index_to_dti([model.times[-1]])
    mob_effect_fig.update_xaxes(range=(plot_start_time, end_date[0]))
    mob_effect_fig.update_layout(showlegend=False)
    mob_effect_fig.write_image(SUPPLEMENT_PATH / mob_effect_filename)
    return mobility_effect, mobility_effect_func, mob_adj_text, mob_effect_fig, mob_effect_filename


def get_mobility_mapper() -> tuple:
    mob_map_text = "The mobility mapping function is used to scale the contribution of contacts at " \
        "workplaces and in `other locations' to the overall time-varying mixing matrix " \
        "(that is, contacts in locations other than the home and in schools). "
    def mobility_scaling(home_matrix, school_contacts, work_matrix, work_scaler, other_matrix, other_scaler):
        return home_matrix + school_contacts + work_matrix * work_scaler + other_matrix * other_scaler
    return mobility_scaling, mob_map_text


def add_age_stratification(
    compartments: list,
    age_strata: list,
    pop_splits: pd.Series,
    matrix: np.array,
) -> tuple:
    """
    Args:
        pop_splits: The proportion of the population to assign to each age group
        matrix: The mixing matrix to apply
    """
    age_strat = Stratification("agegroup", age_strata, compartments)
    assert len(pop_splits) == len(age_strata), "Different number of age group sizes from age strata request"
    age_strat.set_population_split(pop_splits.to_dict())
    age_strat.set_mixing_matrix(matrix)
    description = "We stratified all compartments of the base model " \
        "into sequential age brackets in five year " \
        "bands from age 0 to 4 through to age 70 to 74 " \
        "with a final age band to represent those aged 75 and above. " \
        "These age brackets were chosen to match those used by the POLYMOD survey and so fit with the mixing data available. " \
        "The population distribution by age group was informed by the data from the Australian " \
        "Bureau of Statistics introduced previously. "
    return age_strat, description


def get_strain_stratification(
    compartments: list,
    strain_strata,
) -> tuple:
    strain_strings = [f"{strain.replace('ba', 'BA.')}" for strain in strain_strata]
    compartments_to_stratify = [comp for comp in compartments if comp != "susceptible"]
    strain_strat = StrainStratification("strain", strain_strata, compartments_to_stratify)
    description = f"We stratified the following compartments according to strain: {', '.join(compartments_to_stratify)}, " \
        f"including compartments to represent strains: {', '.join(strain_strings)}. " \
        f"This was implemented using summer's `{StrainStratification.__name__}' class. "
    return strain_strat, description


def seed_vocs(
    model: CompartmentalModel,
) -> str:
    strains = model.stratifications["strain"].strata
    for strain in strains:
        voc_seed_func = Function(
            triangle_wave_func, 
            [
                Time, 
                Parameter(f"{strain}_seed_time"), 
                Parameter("seed_duration"), 
                Parameter("seed_rate"),
            ]
        )
        model.add_importation_flow(
            f"seed_{strain}",
            voc_seed_func,
            "latent",
            dest_strata={"strain": strain},
            split_imports=True,
        )
    return f"Each strain (including the starting {strains[0].replace('ba', 'BA.')} strain) is seeded through " \
        "a step function that allows the introduction of a constant rate of new infectious " \
        "persons into the system over a fixed seeding duration. "


def add_reinfection(
    model: CompartmentalModel,
    strain_strata: list,
    mob_adjuster,
) -> str:
    for dest_strain in strain_strata:
        for source_strain in strain_strata:
            process = "early_reinfection"
            origin = "recovered"
            destination = "latent"
            if int(dest_strain[-1]) > int(source_strain[-1]): 
                model.add_infection_frequency_flow(
                    process, 
                    mob_adjuster * Parameter("contact_rate") * Parameter(f"{dest_strain}_escape"),
                    origin, 
                    destination,
                    source_strata={"strain": source_strain},
                    dest_strata={"strain": dest_strain},
                )
            process = "late_reinfection"
            origin = "waned"
            model.add_infection_frequency_flow(
                process, 
                mob_adjuster * Parameter("contact_rate"),
                origin, 
                destination,
                source_strata={"strain": source_strain},
                dest_strata={"strain": dest_strain},
            )

    return "Reinfection is possible from both the recovered " \
        "and waned compartments, with these processes termed " \
        "`early' and `late' reinfection respectively. " \
        "In the case of early reinfection, this is only possible " \
        "for persons who have recovered from an earlier circulating sub-variant. " \
        "That is, BA.2 early reinfection is possible for persons previously infected with " \
        "BA.1, while BA.5 reinfection is possible for persons previously infected with " \
        "BA.1 or BA.2. The degree of immune escape is determined by the infecting variant " \
        "and differs for BA.2 and BA.5. This implies that the rate of reinfection " \
        "is equal for BA.5 reinfecting those recovered from past BA.1 infection " \
        "as it is for those recovered from past BA.2 infection. " \
        "For late reinfection, all natural immunity is lost for persons in the waned compartment, " \
        "such that the rate of reinfection for these persons is the same as the rate of infection " \
        "for fully susceptible persons. " \
        "As for the first infection process, all reinfection processes transition individuals " \
        "to the latent compartment corresponding to the infecting strain. "


def add_incidence_output(
    model: CompartmentalModel,
    infection_processes: list,
) -> str:
    output = "incidence"
    for process in infection_processes:
        model.request_output_for_flow(f"{process}_onset", process, save_results=False)
    total_infection_processes = sum([DerivedOutput(f"{process}_onset") for process in infection_processes])
    model.request_function_output(output, func=total_infection_processes)
    return f"Modelled {output} is calculated as " \
        f"the absolute rate of {infection_processes[0].replace('_', ' ')} or {infection_processes[1].replace('_', ' ')} " \
        "in the community. "


def get_cdr_values(
    param: float, 
    test_ratios: np.array,
) -> pd.Series:
    return 1.0 - np.exp(0.0 - param * test_ratios)


def add_notifications_output(
    model: CompartmentalModel,
) -> tuple:
    
    # Get data, using test to symptomatic ratio
    hh_impact = load_household_impacts_data()
    hh_test_ratio = hh_impact["Proportion testing"] / hh_impact["Proportion symptomatic"]

    # Do the necessary calculations
    exp_param = get_param_to_exp_plateau(hh_test_ratio[0], Parameter("start_cdr"))
    cdr_values = get_cdr_values(exp_param, hh_test_ratio.to_numpy())

    # Track case detection rate as an interpolated function of time
    aust_epoch = model.get_epoch()
    ratio_interp = get_linear_interpolation_function(
        jnp.array(aust_epoch.datetime_to_number(hh_test_ratio.index)), 
        cdr_values,
    )
    tracked_ratio_interp = model.request_track_modelled_value("ratio_interp", ratio_interp)
    
    # Delay from incidence to notification
    delay = build_gamma_dens_interval_func(Parameter("notifs_shape"), Parameter("notifs_mean"), model.times)

    # Final notification output
    notif_dist_rel_inc = Function(convolve_probability, [DerivedOutput("incidence"), delay]) * tracked_ratio_interp
    model.request_function_output(name="notifications", func=notif_dist_rel_inc)

    survey_fig_name = "survey.jpg"
    survey_fig = hh_impact.plot(labels={"value": "percentage", "index": ""}, markers=True)
    survey_fig.write_image(SUPPLEMENT_PATH / survey_fig_name)
    survey_fig_caption = "Raw survey values."

    ratio_fig_name = "ratio.jpg"
    ratio_fig = hh_test_ratio.plot(labels={"value": "ratio", "index": ""}, markers=True)
    ratio_fig.update_layout(showlegend=False)
    ratio_fig.write_image(SUPPLEMENT_PATH / ratio_fig_name)
    ratio_fig_caption = "Ratio of proportion of households testing to proportion reporting symptoms."

    description = "Notifications are calculated from incidence by first multiplying the raw incidence value " \
        "by a function of time that is intended to capture the declining proportion of COVID-19 episodes " \
        "that were captured through surveillance mechanisms due to a declining proportion of symptomatic " \
        "persons testing over the course of the simulation period, and then applying a convolution." \
        "The Household Impacts of COVID-19 Survey downloaded from " \
        "https://www.abs.gov.au/statistics/people/ people-and-communities/" \
        "household-impacts-covid-19-survey/latest-release on 12th June 2023 " \
        "reports on three indicators, including the proportion of households reporting a household member with symptoms of cold, flu or COVID-19, " \
        "and the proportion of households reporting a household member has had a COVID-19 test. " \
        "The ratio of the second to the first of these indicators was taken as an indicator of declining case detection. " \
        "A transposed exponential function was used to define the relationship between this ratio and the modelled case detection over time, " \
        "with the starting case detection rate varied to capture the uncertainty in the true absolute case detection rate " \
        "proportion of all infection episodes captured through surveillance. " \
        "Specifically, the case detection rate when the ratio is equal to $r$ with starting CDR of $s$ is given by " \
        "$s = (1 - e^{-p \\times r})$. The value of $p$ is calculated to ensure that $s$ is equal to the intended CDR when $r$ is at its starting value. "

    return hh_test_ratio, survey_fig, survey_fig_name, survey_fig_caption, ratio_fig, ratio_fig_name, ratio_fig_caption, description


def track_sero_prevalence(
    compartments: list, 
    model: CompartmentalModel,
) -> str:
    seropos_comps = [comp for comp in compartments if comp != "susceptible"]
    model.request_output_for_compartments("total_pop", compartments)
    model.request_output_for_compartments("seropos", seropos_comps)
    model.request_function_output("seropos_prop", DerivedOutput("seropos") / DerivedOutput("total_pop"))
    return "Seroprevalence is calculated as the proportion of the population ever leaving the susceptible compartment. "


def track_strain_prop(
    strain_strata: list, 
    model: CompartmentalModel,
) -> tuple:
    model.request_output_for_compartments("prev", ["infectious"], save_results=False)
    for strain in strain_strata:
        model.request_output_for_compartments(f"{strain}_prev", ["infectious"], {"strain": strain}, save_results=False)
        model.request_function_output(f"{strain}_prop", DerivedOutput(f"{strain}_prev") / DerivedOutput("prev"))
    return "Proportionate prevalence by strain is tracked as the proportion of the population currently in " \
        "the infectious compartment that is infected with the modelled strain of interest. "


def show_cdr_profiles(
    start_cdr_samples: pd.Series, 
    hh_test_ratio: pd.Series,
) -> tuple:
    """
    Args:
        start_cdr_samples: The CDR parameter values to feed through the algorithm
        hh_test_ratio: The ratio values over time that are fed into the algorithm
    """
    cdr_values = pd.DataFrame()
    for start_cdr in start_cdr_samples:
        exp_param = get_param_to_exp_plateau(hh_test_ratio[0], start_cdr)
        cdr_values[round(start_cdr, 3)] = get_cdr_values(exp_param, hh_test_ratio)

    modelled_cdr_fig_name = "modelled_cdr.jpg"
    modelled_cdr_fig = cdr_values.plot(markers=True, labels={"value": "case detection ratio", "index": ""})
    modelled_cdr_fig.write_image(SUPPLEMENT_PATH / modelled_cdr_fig_name)
    modelled_cdr_fig_caption = "Example case detection rates implemented in randomly selected model runs."

    return modelled_cdr_fig, modelled_cdr_fig_name, modelled_cdr_fig_caption


def show_strain_props(
    strain_strata: list, 
    plot_start_time: datetime.date,
    model: CompartmentalModel,
) -> tuple:
    """
    Args:
        strain_strata: Names of sub-variants
        plot_start_time: Request for left-hand end point for x-axis
        model: Working model

    Returns:
        Output figure, name used to save figure, caption for figure
    """
    
    end_date = model.get_epoch().index_to_dti([model.times[-1]])  # Plot to end of simulation
    strain_props = [f"{strain}_prop" for strain in strain_strata]
    strain_prop_fig_name = "strain_prop.jpg"
    strain_prop_fig = model.get_derived_outputs_df()[strain_props].plot.area(labels={"value": "proportion", "index": ""})
    voc_emerge_df = pd.DataFrame(
        {
            "ba1": [datetime(2021, 11, 22), datetime(2021, 11, 29), datetime(2021, 12, 20), "blue"],
            "ba2": [datetime(2021, 11, 29), datetime(2022, 1, 10), datetime(2022, 3, 7), "red"], 
            "ba5": [datetime(2022, 3, 28), datetime(2022, 5, 16), datetime(2022, 6, 27), "green"],
        },
        index=["any", ">1%", ">50%", "colour"]
    )
    lag = timedelta(days=3.5)  # Dates are given as first day of week in which VoC was first detected
    for voc in voc_emerge_df:
        voc_info = voc_emerge_df[voc]
        colour = voc_info["colour"]
        strain_prop_fig.add_vline(voc_info["any"] + lag, line_dash="dot", line_color=colour)
        strain_prop_fig.add_vline(voc_info[">1%"] + lag, line_dash="dash", line_color=colour)
        strain_prop_fig.add_vline(voc_info[">50%"] + lag, line_color=colour)
    strain_prop_fig.update_xaxes(range=(plot_start_time, end_date[0]))
    strain_prop_fig.update_yaxes(range=(0.0, 1.0))
    strain_prop_fig.write_image(SUPPLEMENT_PATH / strain_prop_fig_name)
    strain_prop_fig_caption = "Proportion of prevalent cases by sub-variant, with first sequence proportion times. " \
        "Dotted line, first isolate of VoC; dashed line, first time VoC represents more than 1% of all isolates; " \
        "solid line, first time VoC represnets more than 50% of all isolates. "
    return strain_prop_fig, strain_prop_fig_name, strain_prop_fig_caption
