from emutools.tex import TexDoc
from .constants import RUN_IDS, PRIMARY_ANALYSIS, BURN_IN, OPTI_DRAWS, ANALYSIS_FEATURES
from aust_covid.constants import RUN_IDS, PRIMARY_ANALYSIS, BURN_IN, OPTI_DRAWS, ANALYSIS_FEATURES


def add_intro_blurb_to_tex(tex_doc: TexDoc):
    description = (
        "The following document describes the methods used in our analyses "
        "of the 2022 SARS-CoV-2 epidemic in Australia. "
        f"We constructed {len(RUN_IDS)} alternative dynamic transmission models "
        "based around the same core features. "
        f"These are named `{', '.join(RUN_IDS.keys())}' and were all based on the features "
        "described in Sections \\ref{base_compartmental_structure}, \\ref{population}, "
        "\\ref{stratification}, \\ref{reinfection}, \\ref{mixing}. "
        f'Two of the models ({", ".join([analysis for analysis, feature in ANALYSIS_FEATURES.items() if feature["mob"]])}) '
        "incorporated additional structure to capture time-varying "
        "mobility \\ref{mobility_extension}, while two incorporated additional structure for time-varying "
        f'vaccination effects {", ".join([analysis for analysis, feature in ANALYSIS_FEATURES.items() if feature["vacc"]])} \\ref{{vaccination_extension}}, '
        "such that these additional features are applied factorially "
        "to the base model structure (therefore including one model with neither extension).\n\n"
        "Each of the four alternative modelling approaches were then calibrated to the "
        "same target data for the 2022 Australian COVID-19 epidemic (see Section \\ref{targets}). "
        "The calibration algorithms were also harmonised to the greatest extent possible (see Section \\ref{calibration_methods}), "
        "although the two analysis approaches that included structure for vaccination "
        "required a different parameter to be substituted for the parameters used "
        "in the analyses not incorporating this structure (as described below). "
        "These four approaches were then compared with regards to their fit to "
        "the target data, with the `mob' analysis (with structure for mobility "
        "but not for vaccination) found to achieve the highest likelihood. "
        f"As a result, this approach to analysis was selected as "
        "the primary analysis (see Section \\ref{analysis_comparison})). "
        "This approach was used for the further analyses, "
        "including parameter inference (e.g. Section \\ref{calibration_results}). "
    )
    tex_doc.add_line(description, "Approach to analyses")


def add_model_structure_blurb_to_tex(tex_doc: TexDoc):
    description = (
        "This section describes the model features that were common to the model "
        "used for all four analyses. "
    )
    tex_doc.add_line(description, "Base compartmental structure")


def add_parameters_blurb_to_tex(tex_doc: TexDoc, cross_ref: bool = True):
    calib_ref = " (See Section \\ref{calibrated_dispersion_parameters})" if cross_ref else ""
    prior_ref = (
        "Calibration priors are identified in the parameters table and illustrated in detail in Section \\ref{priors}. "
        if cross_ref
        else ""
    )
    description = (
        "All epidemiologial parameters, including those used in the calibration algorithm "
        "are presented in Table \\ref{params}. In addition to these epidemiological parameters, "
        "the dispersion parameter for the negative binomial distribution used when calculating "
        "the case time-series and death time-series contributions to the likelihood "
        f"were included in our calibration algorithm {calib_ref}. "
        "The approach to estimating the age-specific infection fatality rate for each "
        "modelled age group is described in \\ref{infection_fatality_rates}. "
        "All epidemiologically significant model parameters were included as priors "
        f"in our calibration algorithm.{prior_ref} "
    )
    tex_doc.add_line(description, "Parameters")


def add_likelihood_blurb_to_tex(tex_doc: TexDoc, cross_ref: bool = True, fig_ref: bool = True):
    target_ref = "Section \\ref{targets})" if cross_ref else "the Targets section"
    fig_ref = (
        "\\ref{case_ranges}, \\ref{death_ranges} and \\ref{seropos_ranges}"
        if fig_ref
        else "presented elsewhere"
    )
    like_ref = " (See Figure \\ref{like_comparison})" if cross_ref else ""
    description = (
        "We compared our four candidate analyses according to their goodness "
        f"of fit to the targets data (described under {target_ref}. "
        "The fit of all four of the models to the target data was considered adequate, "
        f"but the likelihood of the `{PRIMARY_ANALYSIS}' analysis was slightly higher than "
        "that of the other three approaches, with the inclusion of the mobility structure "
        f"appearing to improve the calibration algorithm's fit to targets{like_ref}. "
        f"For this reason, the `{PRIMARY_ANALYSIS}' analysis was considered as the primary analysis "
        "throughout the remaining sections. "
        f"Figures {fig_ref} illustrate "
        "the fit of each candidate model to the target data for "
        "the notification, death and seropositive proportion respectively. "
    )
    tex_doc.add_line(description, "Analysis comparison")


def add_calibration_blurb_to_tex(tex_doc: TexDoc):
    description = (
        "We calibrated our epidemiological model using the `DE Metropolis(Z)' algorithm "
        "provided in the \\href{https://www.pymc.io/welcome.html}{PyMC} package for Bayesian inference.\n\n"
        "Prior to running the algorithm, we ran a short optimisation using "
        "2-point differential evolution provided by Facebook Research's "
        "\\href{https://facebookresearch.github.io/nevergrad/}{nevergrad} library. "
        "Initialisation of the optimisation algorithm was performed using Latin hypercube sampling "
        "to select initial parameter values that were broadly dispersed across the "
        f"multi-dimensional parameter space. The deliberately short optimisation algorithm of {OPTI_DRAWS} draws "
        "was used to move the parameter sets from these dispersed starting positions towards "
        "values with greater likelihood, but remained substantially dispersed from one-another. "
        "\n\nWe then ran the calibration algorithm from these starting points for 10,000 draws "
        "during the tuning phase, and for 50,000 draws during the calibration phase "
        f"of the algorithm, discarding the first {str(int(BURN_IN / 1000))},000 draws as burn-in. "
        "An identical and independent algorithm was applied for each of the four analysis approaches. "
    )
    tex_doc.add_line(description, "Calibration methods")
    description = (
        "The metrics of the performance of our calibration algorithm are presented in "
        "Table \\ref{calibration_metrics}. "
    )
    tex_doc.add_line(description, "Calibration results", subsection="Calibration performance")
    description = (
        "Parameter-specific chain traces with parameter and chain-specific posterior densities "
        f"for the primary `{PRIMARY_ANALYSIS}' analysis "
        "are presented in Figures \\ref{trace_fig_1}, \\ref{trace_fig_2} and \\ref{trace_fig_3}. "
        "These are used for the epidemiological interpretation of our results in the main manuscript. "
        "Overall posterior densitites (pooled over calibration chains) compared against prior distributions are "
        "presented in Figures \\ref{comp_fig_1} and \\ref{comp_fig_2}. "
    )
    tex_doc.add_line(description, "Calibration results", subsection="Parameter inference")


def add_dispersion_blurb_to_tex(tex_doc: TexDoc):
    description = (
        "Figure \\ref{dispersion_examples} provides an illustration of the effect of specific values of the calibrated dispersion "
        "parameter used in the calibration algorithm to adjust the calculation of the contribution "
        "to the likelihood from the notifications and deaths time series. "
    )
    tex_doc.add_line(description, "Targets", subsection="Calibrated dispersion parameters")


def add_mobility_blurb_to_tex(tex_doc: TexDoc):
    description = (
        "In order to consider changes to population mobility over the simulation period, "
        "we scaled the matrix location-specific values according to empiric estimates "
        "of population mobility (as described in the following section). "
        "The school and home locations contributions to the mixing matrix "
        "were kept fixed throughout the simulation. "
        "However, for work and other locations, we used Google mobility estimates "
        "to develop functions of time that scaled the contributions of these locations "
        "to the total values for the matrix. "
        "At a given time point, the product of the scalar function values "
        "with each of these two matrix location components was summed "
        "with the two static location values to obtain the final mixing matrix. "
        "Examples of the final effect of the matrix scaling function on the dynamic "
        "mixing matrices are presented in Figure \\ref{example_matrices}. "
    )
    tex_doc.add_line(description, "Mobility extension", subsection="Application")


def add_vaccination_blurb_to_tex(tex_doc: TexDoc):
    description = (
        "Although Australia's population was relatively unexposed to SARS-CoV-2 "
        "infection and so had little natural immunity at the start of our simulation period, "
        "the population had extensive vaccination-derived immunity across eligible age groups. "
        "This is illustrated in Figure \\ref{full_vacc}, which shows that most age groups "
        "had reached very high coverage with a second-dose of vaccine by early 2022. "
        "As such, we considered that the continuing roll-out of second doses were "
        "very unlikely to have substantially modified the epidemic, "
        "particularly given the questionable impact of such vaccination programs on "
        "onward transmission.\n\n"
        "We therefore considered programs that were rolled out over the course of 2022 "
        "for their potential impact on transmission. "
        "As shown in Figure \\ref{program_coverage}, we considered that the most likely "
        "programs to have had an effect on transmission through 2022 were the fourth dose "
        "`winter booster' program, and the primary course (completing second doses) "
        "program for children aged 5 to 11 years. "
        "The heterogeneous immunity stratification of the base model was utilised to "
        "consider the impact of these programs on community transmission, "
        "as described in the following section. "
    )
    tex_doc.add_line(description, "Vaccination extension", subsection="Rationale")
    description = (
        "Using the various reporting streams from which the vaccination data "
        "were derived (illustrated as different colours in Figure \\ref{program_coverage}), "
        "we calculated the total number of persons vaccinated. "
        "Next, we converted this to a proportion using the population denominators supplied "
        "by the Commonwealth in the same dataset. "
        "We then calculated the proportion of the population previously unvaccinated "
        "under each of these programs and divided by the time interval over which this occurred "
        "to calculate the rate at which these population groups "
        "received vaccination (Figure \\ref{vacc_implement}). "
        "These were then applied as unidirectional flows that transitioned persons from "
        "the unvaccinated stratum to the vaccinated stratum. "
        "For this extended model configuration, a third stratum was added to "
        "the immunity stratification. "
        "The reported coverage and coverage lagged by 14 days are compared against the "
        "modelled population distribution across the three immunity strata in Figure \\ref{vaccination_distribution}. "
    )
    tex_doc.add_line(description, "Vaccination extension", subsection="Application")
