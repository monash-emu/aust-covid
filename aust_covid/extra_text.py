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


def add_likelihood_blurb_to_tex(tex_doc: TexDoc):
    description = 'We compared our four candidate analyses according to their goodness ' \
        'of fit to the targets data (described under Section \\ref{targets}). ' \
        'The fit of all four of the models to the target data was considered adequate, ' \
        f"but the likelihood of the `{PRIMARY_ANALYSIS}' was slightly higher than " \
        'that of the other three approaches, with the inclusion of the mobility structure ' \
        "appearing to improve the calibration algorithm's fit to targets (See Figure \\ref{like_comparison}). " \
        f"For this reason, the `{PRIMARY_ANALYSIS}' was considered as the primary analysis " \
        'throughout the remaining sections. ' \
        'Figures \\ref{case_ranges}, \\ref{death_ranges} and \\ref{seropos_ranges} illustrate ' \
        'the fit of each candidate model to the target data for ' \
        'the notification, death and seropositive proportion respectively. '
    tex_doc.add_line(description, 'Analysis comparison')
