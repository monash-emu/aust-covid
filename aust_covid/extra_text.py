from inputs.constants import RUN_IDS, PRIMARY_ANALYSIS, BURN_IN, OPTI_DRAWS
from emutools.tex import TexDoc


def add_intro_blurb_to_tex(tex_doc: TexDoc):
    description = 'The following document describes the methods used in our analyses ' \
        'of the 2022 SARS-CoV-2 epidemic in Australia. ' \
        f'We constructed {len(RUN_IDS)} alternative dynamic models ' \
        'based around the same core features. ' \
        f"These are named `{', '.join(RUN_IDS.keys())}' and were all based on the features " \
        'described in Sections \\ref{base_compartmental_structure}, \\ref{population}, ' \
        '\\ref{stratification}, \\ref{reinfection}, \\ref{mixing}. ' \
        'Two of the models incorporated additional structure to capture time-varying ' \
        'mobility \\ref{mobility_extension}, while two incorporated additional structure for time-varying ' \
        'vaccination effects \\ref{vaccination_extension}, such that these additional features are applied factorially ' \
        'to the core model structure.\n\n' \
        'Each of the four alternative modelling approaches were then calibrated to the ' \
        'same target data for the 2022 Australian COVID-19 epidemic (see Section \\ref{targets}). ' \
        'The calibration algorithms were also harmonised to the greatest extent possible (see Section \\ref{calibration_methods}), ' \
        'although the two analysis approaches that included structure for vaccination ' \
        'required a different parameter to be substituted for the parameters used ' \
        'in the analyses not incorporating this structure (as described below). ' \
        'These four approaches were then compared with regards to their fit to ' \
        "the target data, with the `mob' analysis (with structure for mobility " \
        'but not for vaccination) found to achieve the highest likelihood. ' \
        f"As a result, this approach to analysis was selected as " \
        'the primary analysis (see Section \\ref{analysis_comparison})). ' \
        'This approach was used for the further analyses, ' \
        'including parameter inference (e.g. Section \\ref{calibration_results}). '
    tex_doc.add_line(description, 'Approach to analyses')


def add_model_structure_blurb_to_tex(tex_doc: TexDoc):
    description = 'This section describes the model features that were common to the model ' \
        'used for all four analyses. '
    tex_doc.add_line(description, 'Base compartmental structure')


def add_parameters_blurb_to_tex(tex_doc: TexDoc):
    description = 'All model parameters, including those used in the calibration algorithm ' \
        'are presented in Table \\ref{params}. ' \
        'The approach to estimating the age-specific infection fatality rate for each ' \
        'modelled age group is described in \\ref{infection_fatality_rates}. ' \
        'All epidemiologically significant model parameters were included as priors ' \
        'in our calibration algorithm. Calibration priors are identified in the parameters table ' \
        'and illustrated in detail in  Section \\ref{priors}. '
    tex_doc.add_line(description, 'Parameters')


def add_likelihood_blurb_to_tex(tex_doc: TexDoc):
    description = 'We compared our four candidate analyses according to their goodness ' \
        'of fit to the targets data (described under Section \\ref{targets}). ' \
        'The fit of all four of the models to the target data was considered adequate, ' \
        f"but the likelihood of the `{PRIMARY_ANALYSIS}' analysis was slightly higher than " \
        'that of the other three approaches, with the inclusion of the mobility structure ' \
        "appearing to improve the calibration algorithm's fit to targets " \
        '(See Figure \\ref{like_comparison}). ' \
        f"For this reason, the `{PRIMARY_ANALYSIS}' was considered as the primary analysis " \
        'throughout the remaining sections. ' \
        'Figures \\ref{case_ranges}, \\ref{death_ranges} and \\ref{seropos_ranges} illustrate ' \
        'the fit of each candidate model to the target data for ' \
        'the notification, death and seropositive proportion respectively. '
    tex_doc.add_line(description, 'Analysis comparison')


def add_calibration_blurb_to_tex(tex_doc: TexDoc):
    description = "We calibrated the model using the `DE Metropolis Z' method " \
        'provided in the \\href{https://www.pymc.io/welcome.html}{PyMC} package for Bayesian inference.\n\n' \
        'First, we used Latin hypercube sampling to select parameter values from across the ' \
        f'multi-dimensional parameter space. Next, we ran a short optimisation algorithm of {OPTI_DRAWS} draws ' \
        "using Facebook Research's \\href{https://facebookresearch.github.io/nevergrad/}{nevergrad}" \
        'to move the parameter sets from these dispersed starting positions towards ' \
        'values that were associated with a greater likelihood, ' \
        'but remained substantially dispersed from one-another. ' \
        'We then ran the calibration algorithm from these starting points for 10,000 draws ' \
        'during the tuning phase, and for 20,000 draws during the calibration phase ' \
        f'of the algorithm, discarding the first {str(BURN_IN)} draws as burn-in. ' \
        'An identical and independent algorithm was applied for each of the four analysis approaches. '        
    tex_doc.add_line(description, 'Calibration methods')
    description = 'The metrics of the performance of our calibration algorithm are presented in ' \
        'Table \\ref{calibration_metrics}. '
    tex_doc.add_line(description, 'Calibration results', subsection='Calibration performance')
    description = 'Parameter-specific chain traces with parameter and chain-specific posterior densities ' \
        f"for the primary `{PRIMARY_ANALYSIS}' analysis " \
        'are presented in Figures \\ref{trace_fig_1}, \\ref{trace_fig_2} and \\ref{trace_fig_3}. ' \
        'These are used for the epidemiological interpretation of our results in the main manuscript. ' \
        'Overall posterior densitites (pooled over calibration chains) compared against prior distributions are ' \
        'presented in Figures \\ref{comp_fig_1} and \\ref{comp_fig_2}. '
    tex_doc.add_line(description, 'Calibration results', subsection='Parameter inference')


def add_dispersion_blurb_to_tex(tex_doc: TexDoc):
    description = 'Figure \\ref{dispersion_examples} provides an illustration of the effect of specific values of the calibrated dispersion ' \
        'parameter used in the calibration algorithm to adjust the calculation of the contribution ' \
        'to the likelihood from the notifications and deaths time series. '
    tex_doc.add_line(description, 'Targets', subsection='Calibrated dispersion parameters')


def add_mobility_blurb_to_tex(tex_doc: TexDoc):
    description = 'The two scaling functions developed in the previous were used to adjust ' \
        'rates of contact over time in the two time-varying matrix locations. ' \
        'These were summed with the two static locations to obtain the final matrix. ' \
        'Examples of the final effect of the matrix scaling function on the dynamic ' \
        'mixing matrices are presented in Figure \\ref{example_matrices}. '
    tex_doc.add_line(description, 'Mobility extension', subsection='Application')


def add_vaccination_blurb_to_tex(tex_doc: TexDoc):
    description = "Although Australia's population was relatively unexposed to SARS-CoV-2 " \
        'infection and so had little natural immunity at the start of our simulation period, ' \
        'the population had extensive vaccination-derived immunity across eligible age groups. ' \
        'This is illustrated in Figure \\ref{full_vacc}, which shows that most age groups ' \
        'had reached very high coverage with a second-dose of vaccine by early 2022. ' \
        'As such, we considered that the continuing roll-out of second doses were ' \
        'very unlikely to have substantially modified the epidemic, ' \
        'particularly given the questionable impact of such vaccination programs on ' \
        'onward transmission.\n\n' \
        'We therefore considered programs that were rolled out over the course of 2022 ' \
        'for their potential impact on transmission. ' \
        'As shown in Figure \\ref{program_coverage}, we considered that the most likely ' \
        'programs to have had an effect on transmission through 2022 were the fourth dose ' \
        "`winter booster' program, and the primary course (completing second doses) " \
        'program for children aged 5 to 11 years. ' \
        'The heterogeneous immunity stratification of the base model was utilised to ' \
        'consider the impact of these programs on community transmission, ' \
        'as described in the following section. '
    tex_doc.add_line(description, 'Vaccination extension', subsection='Rationale')
    description = 'Using the various reporting streams from which the vaccination data ' \
        'were derived (illustrated as different colours in Figure \\ref{program_coverage}), ' \
        'we calculated the total number of persons vaccinated. ' \
        'Next, we converted this to a proportion using the population denominators supplied ' \
        'by the Commonwealth in the same dataset. ' \
        'We then calculated the proportion of the population previously unvaccinated ' \
        'under each of these programs and divided by the time interval over which this occurred ' \
        'to calculate the rate at which these population groups ' \
        'received vaccination (Figure \\ref{vacc_implement}). ' \
        'These were then applied as unidirectional flows taht transitioned persons from ' \
        'the unvaccinated stratum to the vaccinated stratum. ' \
        'For this extended model configuration, a third stratum was added to ' \
        'the immunity stratification. ' \
        'The reported coverage and coverage lagged by 14 days are compared against the ' \
        'modelled population distribution across the three immunity strata in Figure \\ref{vaccination_distribution}. '
    tex_doc.add_line(description, 'Vaccination extension', subsection='Application')
