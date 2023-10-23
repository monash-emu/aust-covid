from inputs.constants import RUN_IDS, PRIMARY_ANALYSIS
from emutools.tex import TexDoc


def add_intro_blurb_to_tex(tex_doc: TexDoc):
    description = 'The following document describes the methods used in our analyses ' \
        'of the 2022 SARS-CoV-2 epidemic in Australia. ' \
        f'We constructed {len(RUN_IDS)} alternative dynamic models ' \
        'based around the same core features. ' \
        f"These were termed `{', '.join(RUN_IDS.keys())}' and were all based on the features " \
        'described in Sections \\ref{base_compartmental_structure}, \\ref{population}, ' \
        '\\ref{stratification}, \\ref{reinfection}, \\ref{mixing}. ' \
        'Two of the models incorporated additional structure to capture time-varying ' \
        'mobility, while two incorporated additional structure for time-varying ' \
        'vaccination effects, such that these additional features are applied factorially ' \
        'to the core model structure.\n\n' \
        'Each of the four alternative modelling approaches were then calibrated to the ' \
        'same target data for the 2022 Australian COVID-19 epidemic. ' \
        'The calibration algorithms were also harmonised to the greatest extent possible, ' \
        'although the two analysis approaches that included structure for vaccination ' \
        'required a different parameter to be substituted for the parameters used ' \
        'in the analyses not incorporating this structure. ' \
        'These four approaches were then compared with regards to their fit to ' \
        'the target data, with the analysis that included structure for mobility ' \
        'but not for vaccination found to achieve the highest likelihood. ' \
        f"As a result of this comparison, the `{PRIMARY_ANALYSIS}' analysiswas  selected as " \
        'the primary analysis (see Section \\ref{analysis_comparison})). ' \
        'This approach was used for the further analyses, ' \
        'including parameter inference (e.g. Section \\ref{calibration_results}). '
    tex_doc.add_line(description, 'Approach to analyses')


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
        '(See Figures \\ref{like_comparison} and \\ref{violin_comparison}). ' \
        f"For this reason, the `{PRIMARY_ANALYSIS}' was considered as the primary analysis " \
        'throughout the remaining sections. ' \
        'Figures \\ref{case_ranges}, \\ref{death_ranges} and \\ref{seropos_ranges} illustrate ' \
        'the fit of each candidate model to the target data for ' \
        'the notification, death and seropositive proportion respectively. '
    tex_doc.add_line(description, 'Analysis comparison')


def add_calibration_blurb_to_tex(tex_doc: TexDoc):
    description = 'The metrics of the performance of our calibration algorithm are presented in ' \
        'Table \\ref{calibration_metrics}. '
    tex_doc.add_line(description, 'Calibration', subsection='Calibration performance')
    description = 'Parameter-specific chain traces with parameter and chain-specific posterior densities ' \
        'are presented in Figures \\ref{trace_fig_1}, \\ref{trace_fig_2} and \\ref{trace_fig_3}. ' \
        'Overall posterior densitites (pooled over calibration chains) compared against prior distributions are ' \
        'presented in Figures \\ref{comp_fig_1} and \\ref{comp_fig_2}. '
    tex_doc.add_line(description, 'Calibration', subsection='Parameter inference')


def add_dispersion_blurb_to_tex(tex_doc: TexDoc):
    description = 'Figure \\ref{dispersion_examples} provides an illustration of the effect of specific values of the calibrated dispersion ' \
        'parameter used in the calibration algorithm to adjust the calculation of the contribution ' \
        'to the likelihood from the notifications and deaths time series. '
    tex_doc.add_line(description, 'Targets')
