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

from aust_covid.doc_utils import TextElement, FigElement, DocumentedProcess
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


class DocumentedAustModel(DocumentedProcess):
    """
    The Australia-specific documented model.
    Constructed through sequentially calling its methods.
    Rather than docstrings for each, the text string to be included 
    in the documentation is best description of the code's function.

    Args:
        DocumentedProcess: General epidemiological process with documentation
    """
    ref_date = datetime(2019, 12, 31)
    compartments = [
        "susceptible",
        "latent",
        "infectious",
        "recovered",
        "waned",
    ]
    age_strata = list(range(0, 75, 5))
    strain_strata = {
        "ba1": "BA.1",
        "ba2": "BA.2",
        "ba5": "BA.5",
    }
    infection_processes = [
        "infection", 
        "early_reinfection",
        "late_reinfection",
    ]

    def __init__(
        self, 
        doc=None, 
        add_documentation=False,
    ):
        super().__init__(doc, add_documentation)

    def build_base_model(
        self,
        start_date: datetime,
        end_date: datetime,
    ):
        infectious_compartment = "infectious"
        self.model = CompartmentalModel(
            times=(
                (start_date - self.ref_date).days, 
                (end_date - self.ref_date).days,
            ),
            compartments=self.compartments,
            infectious_compartments=[infectious_compartment],
            ref_date=self.ref_date,
        )

        if self.add_documentation:
            description = f"The base model consists of {len(self.compartments)} states, " \
                f"representing the following states: {', '.join(self.compartments)}. " \
                f"Only the {infectious_compartment} compartment contributes to the force of infection. " \
                f"The model is run from {str(start_date.date())} to {str(end_date.date())}. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def get_pop_data(self):
        pop_data, sheet_name = load_pop_data()

        if self.add_documentation:
            description = f"For estimates of the Australian population, the {sheet_name} spreadsheet was downloaded "
            description2 = "from the Australian Bureau of Statistics website on the 1st of March 2023 \cite{abs2022}. "
            self.add_element_to_doc("General model construction", TextElement(description))
            self.add_element_to_doc("General model construction", TextElement(description2))

        return pop_data

    def set_model_starting_conditions(
        self, 
        pop_data: pd.DataFrame,
    ):
        total_pop = pop_data["Australia"].sum()
        self.model.set_initial_population({"susceptible": total_pop})
        
        if self.add_documentation:
            description = f"The simulation starts with {str(round(total_pop / 1e6, 3))} million susceptible persons only, " \
                "with infectious persons introduced later through strain seeding as described below. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_infection_to_model(self):
        process = "infection"
        origin = "susceptible"
        destination = "latent"
        self.model.add_infection_frequency_flow(process, Parameter("contact_rate"), origin, destination)
        
        if self.add_documentation:
            description = f"The {process} process moves people from the {origin} " \
                f"compartment to the {destination} compartment, " \
                "under the frequency-dependent transmission assumption. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_progression_to_model(self):
        process = "progression"
        origin = "latent"
        destination = "infectious"
        self.model.add_transition_flow(process, 1.0 / Parameter("latent_period"), origin, destination)

        if self.add_documentation:
            description = f"The {process} process moves " \
                f"people directly from the {origin} state to the {destination} compartment, " \
                "with the rate of transition calculated as the reciprocal of the latent period. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_recovery_to_model(self):
        process = "recovery"
        origin = "infectious"
        destination = "recovered"
        self.model.add_transition_flow(process, 1.0 / Parameter("infectious_period"), origin, destination)

        if self.add_documentation:
            description = f"The {process} process moves " \
                f"people directly from the {origin} state to the {destination} compartment, " \
                "with the rate of transition calculated as the reciprocal of the infectious period. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_waning_to_model(self):
        process = "waning"
        origin = "recovered"
        destination = "waned"

        self.model.add_transition_flow(process, 1.0 / Parameter("high_immunity_period"), origin, destination)

    def add_early_reinfection_to_model(self):
        process = "early_reinfection"
        origin = "recovered"
        destination = "latent"
        for dest_strain in self.strain_strata:
            for source_strain in self.strain_strata:
                if int(dest_strain[-1]) > int(source_strain[-1]): 
                    self.model.add_infection_frequency_flow(
                        process, 
                        Parameter("contact_rate") * Parameter(f"{dest_strain}_escape"),
                        origin, 
                        destination,
                        source_strata={"strain": source_strain},
                        dest_strata={"strain": dest_strain},
                    )
        
        if self.add_documentation:
            description = f"The {process} moves people from the {origin} " \
                f"compartment to the {destination} compartment, " \
                "under the frequency-dependent transmission assumption. " \
                "Reinfection with a later sub-variant is only possible " \
                "for persons who have recovered from an earlier sub-variant. " \
                "That is, BA.2 reinfection is possible for persons previously infected with " \
                "BA.1, while BA.5 reinfection is possible for persons previously infected with " \
                "BA.1 or BA.2. The degree of immune escape is determined by the infecting variant " \
                "and differs for BA.2 and BA.5. This implies that the rate of reinfection " \
                "is equal for BA.5 reinfecting those recovered from past BA.1 infection " \
                "as it is for those recovered from past BA.2 infection. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_late_reinfection_to_model(self):
        process = "late_reinfection"
        origin = "waned"
        destination = "latent"

        self.model.add_infection_frequency_flow(process, Parameter("contact_rate"), origin, destination)

    def add_incidence_output_to_model(self):
        output = "incidence"
        for process in self.infection_processes:
            self.model.request_output_for_flow(f"{process}_onset", process, save_results=False)
        self.model.request_function_output(
            output,
            func=sum([DerivedOutput(f"{process}_onset") for process in self.infection_processes])
        )

        if self.add_documentation:
            description = f"Modelled {output} is calculated as " \
                f"the absolute rate of {self.infection_processes[0]} or {self.infection_processes[1]} " \
                "in the community. "
            self.add_element_to_doc("Outputs", TextElement(description))

    def add_notifications_output_to_model(self):
        output = "notifications"
        output_to_convolve = "incidence"
        delay = build_gamma_dens_interval_func(Parameter("notifs_shape"), Parameter("notifs_mean"), self.model.times)
        notif_dist_rel_inc = Function(convolve_probability, [DerivedOutput(output_to_convolve), delay]) * Parameter("cdr")
        self.model.request_function_output(name=output, func=notif_dist_rel_inc)

        if self.add_documentation:
            description = f"Modelled {output} is calculated as " \
                f"the {output_to_convolve} rate convolved with a gamma-distributed onset to notification delay, " \
                f"multiplied by the case detection rate. "
            self.add_element_to_doc("Outputs", TextElement(description))

    def track_age_specific_incidence(self):
        for age in self.model.stratifications["agegroup"].strata:
            for process in self.infection_processes:
                self.model.request_output_for_flow(
                    f"{process}_onsetXagegroup_{age}", 
                    process, 
                    source_strata={"agegroup": age},
                    save_results=False,
                )
            self.model.request_function_output(
                f"incidenceXagegroup_{age}",
                func=sum([DerivedOutput(f"{process}_onsetXagegroup_{age}") for process in self.infection_processes]),
                save_results=False,
            )
    
    def add_death_output_to_model(self):
        agegroups = self.model.stratifications["agegroup"].strata
        for age in agegroups:
            age_output = f"deathsXagegroup_{age}"
            output_to_convolve = f"incidenceXagegroup_{age}"
            delay = build_gamma_dens_interval_func(Parameter("deaths_shape"), Parameter("deaths_mean"), self.model.times)
            death_dist_rel_inc = Function(convolve_probability, [DerivedOutput(output_to_convolve), delay]) * Parameter(f"ifr_{age}")
            self.model.request_function_output(name=age_output, func=death_dist_rel_inc, save_results=False)
        output = "deaths"
        self.model.request_function_output(
            output,
            func=sum([DerivedOutput(f"deathsXagegroup_{age}") for age in agegroups]),
        )

        if self.add_documentation:
            description = f"Modelled {output} is calculated from " \
                f"the age-specific incidence rate convolved with a gamma-distributed onset to death delay, " \
                f"multiplied by an age-specific infection fatality rate for each age bracket. " \
                f"The time series of deaths for each age gorup is then summed to obtain total modelled {output}. "
            self.add_element_to_doc("Outputs", TextElement(description))

    def build_polymod_britain_matrix(self) -> np.array:
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

        if self.add_documentation:
            description = "We took unadjusted estimates for interpersonal rates of contact by age " \
                "from the United Kingdom data provided by Mossong et al.'s POLYMOD study \cite{mossong2008}. " \
                "The data were obtained from https://doi.org/10.1371/journal.pmed.0050074.st005 " \
                "on 12th February 2023 (downloaded in their native docx format). " \
                "The matrix is transposed because summer assumes that rows represent infectees " \
                "and columns represent infectors, whereas the POLYMOD data are labelled " \
                "`age of contact' for the rows and `age group of participant' for the columns. "
            self.add_element_to_doc("General model construction", TextElement(description))
            
            filename = "raw_matrix.jpg"
            matrix_fig = px.imshow(matrix, x=self.age_strata, y=self.age_strata)
            matrix_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Raw matrices from Great Britain POLYMOD. Values are contacts per person per day. "
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        return matrix

    def adapt_gb_matrix_to_aust(
        self,
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
        modelled_pops.index = self.age_strata
        aust_age_props = pd.Series([pop / aust_pop_series.sum() for pop in modelled_pops], index=self.age_strata)
        assert len(aust_age_props) == unadjusted_matrix.shape[0], "Different number of Aust age groups from mixing categories"

        # UK population distributions
        raw_uk_data = load_uk_pop_data()
        uk_age_pops = raw_uk_data[:14]
        uk_age_pops["70 years and up"] = raw_uk_data[14:].sum()
        uk_age_pops.index = self.age_strata
        uk_age_props = uk_age_pops / uk_age_pops.sum()
        assert len(uk_age_props) == unadjusted_matrix.shape[0], "Different number of UK age groups from mixing categories"
        
        # Calculation
        aust_uk_ratios = aust_age_props / uk_age_props
        adjusted_matrix = np.dot(unadjusted_matrix, np.diag(aust_uk_ratios))
        
        if self.add_documentation:
            description = "Matrices were adjusted to account for the differences in the age distribution of the " \
                "Australian population distribution in 2022 compared to the population of Great Britain in 2000. " \
                "The matrices were adjusted by taking the dot product of the unadjusted matrices and the diagonal matrix " \
                "containing the vector of the ratios between the proportion of the British and Australian populations " \
                "within each age bracket as its diagonal elements. " \
                "To align with the methodology of the POLYMOD study \cite{mossong2008} " \
                "we sourced the 2001 UK census population for those living in the UK at the time of the census " \
                "from the Eurostat database (https://ec.europa.eu/eurostat). "
            self.add_element_to_doc("Age stratification", TextElement(description))

            filename = "input_population.jpg"
            pop_fig = px.bar(aust_pop_series, labels={"value": "population", "Age (years)": ""})
            pop_fig.update_layout(showlegend=False)
            pop_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Population sizes by age group obtained from Australia Bureau of Statistics."
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

            filename = "modelled_population.jpg"
            modelled_pop_fig = px.bar(modelled_pops, labels={"value": "population", "index": ""})
            modelled_pop_fig.update_layout(showlegend=False)
            modelled_pop_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Population sizes by age group included in the model."
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

            filename = "matrix_ref_pop.jpg"
            uk_pop_fig = px.bar(uk_age_pops, labels={"value": "population", "index": ""})
            uk_pop_fig.update_layout(showlegend=False)
            uk_pop_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "United Kingdom population sizes."
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

            filename = "adjusted_matrix.jpg"
            matrix_plotly_fig = px.imshow(unadjusted_matrix, x=self.age_strata, y=self.age_strata)
            matrix_plotly_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Matrices adjusted to Australian population. Values are contacts per person per day. "
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        aust_age_props.index = aust_age_props.index.astype(str)
        return adjusted_matrix, aust_age_props

    def add_age_stratification_to_model(
        self,
        pop_splits: pd.Series,
        matrix: np.array,
    ):
        """
        Args:
            pop_splits: The proportion of the population to assign to each age group
            matrix: The mixing matrix to apply
        """

        age_strat = Stratification("agegroup", self.age_strata, self.compartments)
        assert len(pop_splits) == len(self.age_strata), "Different number of age group sizes from age strata request"
        age_strat.set_population_split(pop_splits.to_dict())
        age_strat.set_mixing_matrix(matrix)
        self.model.stratify_with(age_strat)

        if self.add_documentation:
            description = "We stratified all compartments of the base model " \
                "into sequential age brackets in five year " \
                "bands from age 0 to 4 through to age 65 to 69 " \
                "with a final age band to represent those aged 70 and above. " \
                "These age brackets were chosen to match those used by the POLYMOD survey. " \
                "The population distribution by age group was informed by the data from the Australian " \
                "Bureau of Statistics introduced previously. "
            self.add_element_to_doc("Age stratification", TextElement(description))
        
    def get_strain_stratification(self) -> StrainStratification:
        strain_strings = list(self.strain_strata.keys())
        compartments_to_stratify = [comp for comp in self.compartments if comp != "susceptible"]
        strain_strat = StrainStratification("strain", strain_strings, compartments_to_stratify)

        if self.add_documentation:
            description = f"We stratified the following compartments according to strain: {', '.join(compartments_to_stratify)}. " \
                f"including compartments to represent strains: {', '.join(self.strain_strata.values())}. " \
                f"This was implemented using summer's `{StrainStratification.__name__}' class. "
            self.add_element_to_doc("Strain stratification", TextElement(description))

        return strain_strat

    def seed_vocs(self):
        strains = self.model.stratifications["strain"].strata
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
            self.model.add_importation_flow(
                "seed_{strain}",
                voc_seed_func,
                "latent",
                dest_strata={"strain": strain},
                split_imports=True,
            )

        if self.add_documentation:
            description = f"Each strain (including the starting {strains[0]} strain) is seeded through " \
                "a step function that allows the introduction of a constant rate of new infectious " \
                "persons into the system over a fixed seeding duration. "
            self.add_element_to_doc("Strain stratification", TextElement(description))


def build_aust_model(
    start_date: datetime,
    end_date: datetime,
    doc: pl.document.Document,
    add_documentation: bool=False,
) -> CompartmentalModel:
    """
    Build a fairly basic model, as described in the component functions called.
    
    Returns:
        The model object
    """

    # Basic model construction
    aust_model = DocumentedAustModel(doc, add_documentation)
    pop_data = aust_model.get_pop_data()
    aust_model.build_base_model(start_date, end_date)
    aust_model.set_model_starting_conditions(pop_data)
    aust_model.add_infection_to_model()
    aust_model.add_progression_to_model()
    aust_model.add_recovery_to_model()
    aust_model.add_waning_to_model()

    # Age stratification
    raw_matrix = aust_model.build_polymod_britain_matrix()
    adjusted_matrix, pop_splits = aust_model.adapt_gb_matrix_to_aust(raw_matrix, pop_data)
    aust_model.add_age_stratification_to_model(pop_splits, adjusted_matrix)
    
    # Strain stratification
    aust_model.model.stratify_with(aust_model.get_strain_stratification())
    aust_model.seed_vocs()

    # Reinfection (must come after strain stratification)
    aust_model.add_early_reinfection_to_model()
    aust_model.add_late_reinfection_to_model()

    # Outputs (must come after infection and reinfection)
    aust_model.add_incidence_output_to_model()
    aust_model.add_notifications_output_to_model()
    aust_model.track_age_specific_incidence()
    aust_model.add_death_output_to_model()

    # Documentation
    if add_documentation:
        aust_model.compile_doc()

    return aust_model.model
