import pylatex as pl
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from jax import numpy as jnp
from jax import scipy as jsp

from summer2 import CompartmentalModel, Stratification, StrainStratification
from summer2.parameters import Parameter, DerivedOutput, Function, Time, Data

from aust_covid.inputs import load_pop_data, load_uk_pop_data


BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


def triangle_wave_func(
    time: float, 
    start: float, 
    duration: float, 
    peak: float,
) -> float:
    """
    Generate a peaked triangular wave function
    that starts from and returns to zero.

    Args:
        time: Model time
        start: Time at which wave starts
        duration: Duration of wave
        peak: Peak flow rate for wave
    Returns:
        The wave function
    """
    gradient = peak / (duration * 0.5)
    peak_time = start + duration * 0.5
    time_from_peak = jnp.abs(peak_time - time)
    return jnp.where(time_from_peak < duration * 0.5, peak - time_from_peak * gradient, 0.0)


def convolve_probability(source_output, density_kernel):
    return jnp.convolve(source_output, density_kernel)[:len(source_output)]


def gamma_cdf(
    shape: float, 
    mean: float, 
    x: jnp.array,
) -> jnp.array:
    """
    The regularised gamma function is the CDF of the gamma distribution
    (which is referred to by scipy as "gammainc")

    Args:
        shape: Shape parameter to the desired gamma distribution
        mean: Expectation of the desired gamma distribution
        x: Values to calculate the result over

    Returns:
        Array of CDF values corresponding to input request (x)
    """
    return jsp.special.gammainc(shape, x * shape / mean)


def build_gamma_dens_interval_func(shape, mean, model_times):
    lags = Data(model_times - model_times[0])
    cdf_values = Function(gamma_cdf, [shape, mean, lags])
    return Function(jnp.gradient, [cdf_values])


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""


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
        f"The model is run from {str(start_date.date())} to {str(end_date.date())}. " \
        f"The base model consists of {len(compartments)} states, " \
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
    description = f"For estimates of the Australian population, the {sheet_name} spreadsheet was downloaded "
    description2 = "from the Australian Bureau of Statistics website on the 1st of March 2023 \cite{abs2022}. "
    return pop_data, description + description2


def set_model_starting_conditions(
    model,
    pop_data: pd.DataFrame,
    add_documentation: bool=False
):
    total_pop = pop_data["Australia"].sum()
    model.set_initial_population({"susceptible": total_pop})
    
    if add_documentation:
        description = f"The simulation starts with {str(round(total_pop / 1e6, 3))} million susceptible persons only, " \
            "with infectious persons introduced later through strain seeding as described below. "
        # add_element_to_doc("General model construction", TextElement(description))


def add_infection_to_model(
    model,
    add_documentation: bool=False
):
    process = "infection"
    origin = "susceptible"
    destination = "latent"
    model.add_infection_frequency_flow(process, Parameter("contact_rate"), origin, destination)
    
    if add_documentation:
        description = f"The {process} process moves people from the {origin} " \
            f"compartment to the {destination} compartment, " \
            "under the frequency-dependent transmission assumption. "
        # add_element_to_doc("General model construction", TextElement(description))


def add_progression_to_model(
    model,
    add_documentation: bool=False
):
    process = "progression"
    origin = "latent"
    destination = "infectious"
    model.add_transition_flow(process, 1.0 / Parameter("latent_period"), origin, destination)

    if add_documentation:
        description = f"The {process} process moves " \
            f"people directly from the {origin} state to the {destination} compartment, " \
            "with the rate of transition calculated as the reciprocal of the latent period. "
        # add_element_to_doc("General model construction", TextElement(description))


def add_recovery_to_model(
    model,
    add_documentation: bool=False
):
    process = "recovery"
    origin = "infectious"
    destination = "recovered"
    model.add_transition_flow(process, 1.0 / Parameter("infectious_period"), origin, destination)

    if add_documentation:
        description = f"The {process} process moves " \
            f"people directly from the {origin} state to the {destination} compartment, " \
            "with the rate of transition calculated as the reciprocal of the infectious period. "
        # add_element_to_doc("General model construction", TextElement(description))


def add_waning_to_model(
    model,
    add_documentation: bool=False
):
    process = "waning"
    origin = "recovered"
    destination = "waned"
    model.add_transition_flow(process, 1.0 / Parameter("natural_immunity_period"), origin, destination)

    if add_documentation:
        description = "A waned compartment is included in the model " \
            "to represent persons who no longer have immunity from past natural immunity. " \
            f"Modelled individuals transition from the {origin} compartment to the " \
            f"{destination} compartment at a rate equal to the reciprocal of the " \
            "requested period of time spent with natural immunity. "
        # add_element_to_doc("General model construction", TextElement(description))


def add_reinfection_to_model(
    model,
    strain_strata,
    add_documentation: bool=False
):
    for dest_strain in strain_strata:
        for source_strain in strain_strata:
            process = "early_reinfection"
            origin = "recovered"
            destination = "latent"
            if int(dest_strain[-1]) > int(source_strain[-1]): 
                model.add_infection_frequency_flow(
                    process, 
                    Parameter("contact_rate") * Parameter(f"{dest_strain}_escape"),
                    origin, 
                    destination,
                    source_strata={"strain": source_strain},
                    dest_strata={"strain": dest_strain},
                )
            process = "late_reinfection"
            origin = "waned"
            model.add_infection_frequency_flow(
                process, 
                Parameter("contact_rate"), 
                origin, 
                destination,
                source_strata={"strain": source_strain},
                dest_strata={"strain": dest_strain},
            )

    if add_documentation:
        description = "Reinfection is possible from both the recovered " \
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
            "As for the first infection process, " \
            "all reinfection processes transition people to the model's latent compartment. "
        # add_element_to_doc("General model construction", TextElement(description))


def build_polymod_britain_matrix(
    add_documentation: bool=False
) -> np.array:
    """
    Returns:
        15 by 15 matrix with daily contact rates for age groups
    """

    values = [
        [1.92, 0.65, 0.41, 0.24, 0.46, 0.73, 0.67, 0.83, 0.24, 0.22, 0.36, 0.20, 0.20, 0.26, 0.13],
        [0.95, 6.64, 1.09, 0.73, 0.61, 0.75, 0.95, 1.39, 0.90, 0.16, 0.30, 0.22, 0.50, 0.48, 0.20],
        [0.48, 1.31, 6.85, 1.52, 0.27, 0.31, 0.48, 0.76, 1.00, 0.69, 0.32, 0.44, 0.27, 0.41, 0.33],
        [0.33, 0.34, 1.03, 6.71, 1.58, 0.73, 0.42, 0.56, 0.85, 1.16, 0.70, 0.30, 0.20, 0.48, 0.63],
        [0.45, 0.30, 0.22, 0.93, 2.59, 1.49, 0.75, 0.63, 0.77, 0.87, 0.88, 0.61, 0.53, 0.37, 0.33],
        [0.79, 0.66, 0.44, 0.74, 1.29, 1.83, 0.97, 0.71, 0.74, 0.85, 0.88, 0.87, 0.67, 0.74, 0.33],
        [0.97, 1.07, 0.62, 0.50, 0.88, 1.19, 1.67, 0.89, 1.02, 0.91, 0.92, 0.61, 0.76, 0.63, 0.27],
        [1.02, 0.98, 1.26, 1.09, 0.76, 0.95, 1.53, 1.50, 1.32, 1.09, 0.83, 0.69, 1.02, 0.96, 0.20],
        [0.55, 1.00, 1.14, 0.94, 0.73, 0.88, 0.82, 1.23, 1.35, 1.27, 0.89, 0.67, 0.94, 0.81, 0.80],
        [0.29, 0.54, 0.57, 0.77, 0.97, 0.93, 0.57, 0.80, 1.32, 1.87, 0.61, 0.80, 0.61, 0.59, 0.57],
        [0.33, 0.38, 0.40, 0.41, 0.44, 0.85, 0.60, 0.61, 0.71, 0.95, 0.74, 1.06, 0.59, 0.56, 0.57],
        [0.31, 0.21, 0.25, 0.33, 0.39, 0.53, 0.68, 0.53, 0.55, 0.51, 0.82, 1.17, 0.85, 0.85, 0.33],
        [0.26, 0.25, 0.19, 0.24, 0.19, 0.34, 0.40, 0.39, 0.47, 0.55, 0.41, 0.78, 0.65, 0.85, 0.57],
        [0.09, 0.11, 0.12, 0.20, 0.19, 0.22, 0.13, 0.30, 0.23, 0.13, 0.21, 0.28, 0.36, 0.70, 0.60],
        [0.14, 0.15, 0.21, 0.10, 0.24, 0.17, 0.15, 0.41, 0.50, 0.71, 0.53, 0.76, 0.47, 0.74, 1.47],
    ]

    matrix = np.array(values).T  # Transpose

    if add_documentation:
        description = "We took unadjusted estimates for interpersonal rates of contact by age " \
            "from the United Kingdom data provided by Mossong et al.'s POLYMOD study \cite{mossong2008}. " \
            "The data were obtained from https://doi.org/10.1371/journal.pmed.0050074.st005 " \
            "on 12th February 2023 (downloaded in their native docx format). " \
            "The matrix is transposed because summer assumes that rows represent infectees " \
            "and columns represent infectors, whereas the POLYMOD data are labelled " \
            "`age of contact' for the rows and `age group of participant' for the columns. "
        # add_element_to_doc("General model construction", TextElement(description))
        
        filename = "raw_matrix.jpg"
        matrix_fig = px.imshow(matrix, x=self.age_strata, y=self.age_strata)
        matrix_fig.write_image(SUPPLEMENT_PATH / filename)
        caption = "Raw matrices from Great Britain POLYMOD. Values are contacts per person per day. "
        # add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

    return matrix


def adapt_gb_matrix_to_aust(
    age_strata,
    unadjusted_matrix: np.array, 
    pop_data: pd.DataFrame,
    add_documentation: bool=False
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
    
    if add_documentation:
        description = "Matrices were adjusted to account for the differences in the age distribution of the " \
            "Australian population distribution in 2022 compared to the population of Great Britain in 2000. " \
            "The matrices were adjusted by taking the dot product of the unadjusted matrices and the diagonal matrix " \
            "containing the vector of the ratios between the proportion of the British and Australian populations " \
            "within each age bracket as its diagonal elements. " \
            "To align with the methodology of the POLYMOD study \cite{mossong2008} " \
            "we sourced the 2001 UK census population for those living in the UK at the time of the census " \
            "from the Eurostat database (https://ec.europa.eu/eurostat). "
        # add_element_to_doc("Age stratification", TextElement(description))

        filename = "input_population.jpg"
        pop_fig = px.bar(aust_pop_series, labels={"value": "population", "Age (years)": ""})
        pop_fig.update_layout(showlegend=False)
        pop_fig.write_image(SUPPLEMENT_PATH / filename)
        caption = "Population sizes by age group obtained from Australia Bureau of Statistics."
        # add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        filename = "modelled_population.jpg"
        modelled_pop_fig = px.bar(modelled_pops, labels={"value": "population", "index": ""})
        modelled_pop_fig.update_layout(showlegend=False)
        modelled_pop_fig.write_image(SUPPLEMENT_PATH / filename)
        caption = "Population sizes by age group included in the model."
        # add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        filename = "matrix_ref_pop.jpg"
        uk_pop_fig = px.bar(uk_age_pops, labels={"value": "population", "index": ""})
        uk_pop_fig.update_layout(showlegend=False)
        uk_pop_fig.write_image(SUPPLEMENT_PATH / filename)
        caption = "United Kingdom population sizes."
        # add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        filename = "adjusted_matrix.jpg"
        matrix_plotly_fig = px.imshow(unadjusted_matrix, x=self.age_strata, y=self.age_strata)
        matrix_plotly_fig.write_image(SUPPLEMENT_PATH / filename)
        caption = "Matrices adjusted to Australian population. Values are contacts per person per day. "
        # add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

    aust_age_props.index = aust_age_props.index.astype(str)
    return adjusted_matrix, aust_age_props


def add_incidence_output_to_model(
    model,
    infection_processes,
    add_documentation: bool=False
):
    output = "incidence"
    for process in infection_processes:
        model.request_output_for_flow(f"{process}_onset", process, save_results=False)
    model.request_function_output(
        output,
        func=sum([DerivedOutput(f"{process}_onset") for process in infection_processes])
    )

    if add_documentation:
        description = f"Modelled {output} is calculated as " \
            f"the absolute rate of {infection_processes[0]} or {infection_processes[1]} " \
            "in the community. "
        # add_element_to_doc("Outputs", TextElement(description))


def add_age_stratification_to_model(
    model,
    compartments,
    age_strata,
    pop_splits: pd.Series,
    matrix: np.array,
    add_documentation: bool=False
):
    """
    Args:
        pop_splits: The proportion of the population to assign to each age group
        matrix: The mixing matrix to apply
    """

    age_strat = Stratification("agegroup", age_strata, compartments)
    assert len(pop_splits) == len(age_strata), "Different number of age group sizes from age strata request"
    age_strat.set_population_split(pop_splits.to_dict())
    age_strat.set_mixing_matrix(matrix)
    model.stratify_with(age_strat)

    if add_documentation:
        description = "We stratified all compartments of the base model " \
            "into sequential age brackets in five year " \
            "bands from age 0 to 4 through to age 65 to 69 " \
            "with a final age band to represent those aged 70 and above. " \
            "These age brackets were chosen to match those used by the POLYMOD survey. " \
            "The population distribution by age group was informed by the data from the Australian " \
            "Bureau of Statistics introduced previously. "
        # add_element_to_doc("Age stratification", TextElement(description))


def get_strain_stratification(
    compartments,
    strain_strata,
    add_documentation: bool=False
) -> StrainStratification:
    strain_strings = list(strain_strata.keys())
    compartments_to_stratify = [comp for comp in compartments if comp != "susceptible"]
    strain_strat = StrainStratification("strain", strain_strings, compartments_to_stratify)

    if add_documentation:
        description = f"We stratified the following compartments according to strain: {', '.join(compartments_to_stratify)}. " \
            f"including compartments to represent strains: {', '.join(strain_strata.values())}. " \
            f"This was implemented using summer's `{StrainStratification.__name__}' class. "
        # add_element_to_doc("Strain stratification", TextElement(description))

    return strain_strat


def seed_vocs(
    model,
    add_documentation: bool=False
):
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

    if add_documentation:
        description = f"Each strain (including the starting {strains[0]} strain) is seeded through " \
            "a step function that allows the introduction of a constant rate of new infectious " \
            "persons into the system over a fixed seeding duration. "
        # add_element_to_doc("Strain stratification", TextElement(description))


def add_notifications_output_to_model(
    model,
    add_documentation: bool=False
):
    output = "notifications"
    output_to_convolve = "incidence"
    delay = build_gamma_dens_interval_func(Parameter("notifs_shape"), Parameter("notifs_mean"), model.times)
    notif_dist_rel_inc = Function(convolve_probability, [DerivedOutput(output_to_convolve), delay]) * Parameter("cdr")
    model.request_function_output(name=output, func=notif_dist_rel_inc)

    if add_documentation:
        description = f"Modelled {output} is calculated as " \
            f"the {output_to_convolve} rate convolved with a gamma-distributed onset to notification delay, " \
            f"multiplied by the case detection rate. "
        # add_element_to_doc("Outputs", TextElement(description))


def track_age_specific_incidence(
    model,
    infection_processes,
):
    for age in model.stratifications["agegroup"].strata:
        for process in infection_processes:
            model.request_output_for_flow(
                f"{process}_onsetXagegroup_{age}", 
                process, 
                source_strata={"agegroup": age},
                save_results=False,
            )
        model.request_function_output(
            f"incidenceXagegroup_{age}",
            func=sum([DerivedOutput(f"{process}_onsetXagegroup_{age}") for process in infection_processes]),
            save_results=False,
        )


def add_death_output_to_model(
    model,
    add_documentation: bool=False
):
    agegroups = model.stratifications["agegroup"].strata
    for age in agegroups:
        age_output = f"deathsXagegroup_{age}"
        output_to_convolve = f"incidenceXagegroup_{age}"
        delay = build_gamma_dens_interval_func(Parameter("deaths_shape"), Parameter("deaths_mean"), model.times)
        death_dist_rel_inc = Function(convolve_probability, [DerivedOutput(output_to_convolve), delay]) * Parameter(f"ifr_{age}")
        model.request_function_output(name=age_output, func=death_dist_rel_inc, save_results=False)
    output = "deaths"
    model.request_function_output(
        output,
        func=sum([DerivedOutput(f"deathsXagegroup_{age}") for age in agegroups]),
    )

    if add_documentation:
        description = f"Modelled {output} is calculated from " \
            f"the age-specific incidence rate convolved with a gamma-distributed onset to death delay, " \
            f"multiplied by an age-specific infection fatality rate for each age bracket. " \
            f"The time series of deaths for each age gorup is then summed to obtain total modelled {output}. "
        # add_element_to_doc("Outputs", TextElement(description))
