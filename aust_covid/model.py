import pylatex as pl
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from jax import numpy as jnp

from summer2 import CompartmentalModel, Stratification, StrainStratification
from summer2.parameters import Parameter, DerivedOutput, Function, Time

from aust_covid.doc_utils import TextElement, FigElement, DocumentedProcess


REF_DATE = datetime(2019, 12, 31)
BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


def triangle_wave_func(
        time: Time, 
        start: float, 
        duration: float, 
        peak: float,
    ) -> jnp.where:
    """
    Generate a peaked triangular wave function.

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


def load_pop_data() -> tuple:
    """
    Get the Australian population data from ABS source.

    Returns:
        The population data
        The name of the sheet
    """
    skip_rows = list(range(0, 4)) + list(range(5, 227)) + list(range(328, 332))
    for group in range(16):
        skip_rows += list(range(228 + group * 6, 233 + group * 6))
    sheet_name = "31010do002_202206.xlsx"
    data = pd.read_excel(DATA_PATH / sheet_name, sheet_name="Table_7", skiprows=skip_rows, index_col=[0])
    return data, sheet_name


class DocumentedAustModel(DocumentedProcess):
    """
    The Australia-specific documented model.
    Constructed through sequentially calling its methods.
    Rather than docstrings for each, the text string to be included 
    in the documentation is best description of the code's function.

    Args:
        DocumentedProcess: General epidemiological process with documentation
    """

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
        compartments: list,
    ):
        infectious_compartment = "infectious"
        self.model = CompartmentalModel(
            times=(
                (start_date - REF_DATE).days, 
                (end_date - REF_DATE).days,
            ),
            compartments=compartments,
            infectious_compartments=[infectious_compartment],
            ref_date=REF_DATE,
        )

        if self.add_documentation:
            description = f"The base model consists of {len(compartments)} states, " \
                f"representing the following states: {', '.join(compartments)}. " \
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
                "with infectious persons then introduced through strain seeding as described below. "
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

    def add_reinfection_to_model(
        self, 
        strain_strata: list,
    ):
        process = "reinfection"
        origin = "recovered"
        destination = "latent"
        for dest_strain in strain_strata:
            for source_strain in strain_strata:
                infection_adjuster = 1.0 - Parameter(f"{source_strain}infection_protect_{dest_strain}")
                self.model.add_infection_frequency_flow(
                    process, 
                    Parameter("contact_rate") * infection_adjuster,
                    origin, 
                    destination,
                    source_strata={"strain": source_strain},
                    dest_strata={"strain": dest_strain},
                )
        
        if self.add_documentation:
            description = f"The {process} moves people from the {origin} " \
                f"compartment to the {destination} compartment, " \
                "under the frequency-dependent transmission assumption. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_notifications_output_to_model(self):
        process = "onset"
        output = "notifications"
        transition = "infection"
        self.model.request_output_for_flow(process, transition, save_results=False)
        self.model.request_function_output(output, func=DerivedOutput(process) * Parameter("cdr"))

        if self.add_documentation:
            description = f"Modelled {output} are calculated as " \
                f"the absolute rate of {transition} in the community " \
                "multiplied by the case detection rate. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def build_polymod_britain_matrix(
        self,
        strata: list,
    ) -> np.array:
        """
        Args:
            strata: The strata to apply in age stratification
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
            matrix_fig = px.imshow(matrix, x=strata, y=strata)
            matrix_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Raw matrices from Great Britain POLYMOD. Values are contacts per person per day. "
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        return matrix

    def adapt_gb_matrix_to_aust(
        self,
        unadjusted_matrix: np.array, 
        pop_data: pd.DataFrame,
        strata: list, 
    ) -> tuple:
        """
        Args:
            unadjusted_matrix: The unadjusted matrix
            pop_data: ABS population numbers
            strata: The strata to apply in age stratification
        Returns:
            Matrix adjusted to target population
            Proportions of Australian population in modelled age groups
        """
        
        assert unadjusted_matrix.shape[0] == unadjusted_matrix.shape[1], "Unadjusted mixing matrix not square"

        # UK population distributions
        uk_pops_list = [
            3458060, 3556024, 3824317, 3960916, 3911291, 3762213, 4174675, 4695853, 
            4653082, 3986098, 3620216, 3892985, 3124676, 2706365, 6961183,
        ]
        uk_age_pops = pd.Series(uk_pops_list, index=strata)
        uk_age_props = uk_age_pops / uk_age_pops.sum()
        assert len(uk_age_props) == unadjusted_matrix.shape[0], "Different number of UK age groups from mixing categories"
        
        # Australian population distribution by age        
        aust_pop_series = pop_data["Australia"]
        modelled_pops = pd.concat([aust_pop_series[:"65-69"], pd.Series({"70": aust_pop_series["70-74":].sum()})])
        aust_age_props = pd.Series([pop / aust_pop_series.sum() for pop in modelled_pops], index=strata)
        assert len(aust_age_props) == unadjusted_matrix.shape[0], "Different number of Aust age groups from mixing categories"

        # Calculation
        aust_uk_ratios = aust_age_props / uk_age_props
        adjusted_matrix = np.dot(unadjusted_matrix, np.diag(aust_uk_ratios))
        
        if self.add_documentation:
            description = "Matrices were adjusted to account for the differences in the age distribution of the " \
                "Australian population distribution in 2022 compared to the population of Great Britain in 2008. " \
                "The matrices were adjusted by taking the dot product of the unadjusted matrices and the diagonal matrix " \
                "containing the vector of the ratios between the proportion of the British and Australian populations " \
                "within each age bracket as its diagonal elements. "
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

            filename = "adjusted_matrix.jpg"
            matrix_plotly_fig = px.imshow(unadjusted_matrix, x=strata, y=strata)
            matrix_plotly_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Matrices adjusted to Australian population. Values are contacts per person per day. "
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        aust_age_props.index = aust_age_props.index.astype(str)
        return adjusted_matrix, aust_age_props

    def add_age_stratification_to_model(
        self,
        compartments: list,
        strata: list,
        pop_splits: pd.Series,
        matrix: np.array,
    ):
        """
        Args:
            compartments: All the unstratified model compartments
            strata: The strata to apply
            pop_splits: The proportion of the population to assign to each age group
            matrix: The mixing matrix to apply
        """

        age_strat = Stratification("agegroup", strata, compartments)
        assert len(pop_splits) == len(strata), "Different number of age group sizes from age strata request"
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
        
    def get_strain_stratification(
        self, 
        compartments: list,
        strains: list,
    ) -> tuple:
        """
        Args:
            compartments: Unstratified model compartments
            strains: The names of the strains to use
        """
        strain_strings = list(strains.keys())

        # The strains we're working with
        starting_strain = strain_strings[0]  # BA.1
        other_strains = strain_strings[1:]  # The others, currently just BA.2

        # The stratification object
        compartments_to_stratify = [comp for comp in compartments if comp != "susceptible"]
        starting_compartment = compartments_to_stratify[0]
        strain_strat = StrainStratification("strain", strain_strings, compartments_to_stratify)

        # The starting population split
        population_split = {starting_strain: 1.0}
        population_split.update({strain: 0.0 for strain in other_strains})
        strain_strat.set_population_split(population_split)

        if self.add_documentation:
            description = f"We stratified the following compartments according to strain: {', '.join(compartments_to_stratify)}. " \
                f"including compartments to represent strains: {', '.join(strains.values())}. " \
                f"This was implemented using summer's `{StrainStratification.__name__}' class. " \
                f"All of the starting infectious seed was assigned to the {strains[starting_strain]} category " \
                f"within the {starting_compartment} category. "
            self.add_element_to_doc("Strain stratification", TextElement(description))

        return strain_strat, starting_strain, other_strains

    def seed_vocs(self):
        for strain in self.model.stratifications["strain"].strata:
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
            description = "Each strain (including the starting wild-type) is seeded through " \
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
    compartments = [
        "susceptible",
        "latent",
        "infectious",
        "recovered",
    ]
    aust_model = DocumentedAustModel(doc, add_documentation)
    pop_data = aust_model.get_pop_data()
    aust_model.build_base_model(start_date, end_date, compartments)
    aust_model.set_model_starting_conditions(pop_data)
    aust_model.add_infection_to_model()
    aust_model.add_progression_to_model()
    aust_model.add_recovery_to_model()
    aust_model.add_notifications_output_to_model()

    # Age stratification
    age_strata = list(range(0, 75, 5))
    matrix = aust_model.build_polymod_britain_matrix(age_strata)
    adjusted_matrix, pop_splits = aust_model.adapt_gb_matrix_to_aust(matrix, pop_data, age_strata)
    aust_model.add_age_stratification_to_model(compartments, age_strata, pop_splits, adjusted_matrix)
    
    # Strain stratification
    strain_strata = {
        "ba1": "BA.1",
        "ba2": "BA.2",
    }
    strain_strat, starting_strain, other_strains = aust_model.get_strain_stratification(compartments, strain_strata)
    aust_model.model.stratify_with(strain_strat)
    aust_model.seed_vocs()

    # Reinfection
    aust_model.add_reinfection_to_model(strain_strata)

    # Documentation
    if add_documentation:
        aust_model.compile_doc()

    return aust_model.model
