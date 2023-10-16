from inputs.constants import RUN_IDS, PRIMARY_ANALYSIS
from emutools.tex import TexDoc


def add_intro_blurb_to_tex(tex_doc: TexDoc):
    description = 'The following document describes the methods used in our analyses ' \
        'of the 2022 SARS-CoV-2 epidemic in Australia. ' \
        f'We constructed {len(RUN_IDS)} alternative transmission dynamic models ' \
        'based around similar core features. ' \
        f"These were termed `{', '.join(RUN_IDS.keys())}' and were all based on the features " \
        'described in the following sections. ' \
        'Two of the models incorporated additional structure to capture time-varying ' \
        'mobility, while two incorporated additional structure for time-varying ' \
        'vaccination effects, such that these additional features are applied factorially ' \
        'to the core model structure. ' \
        'Each of the four alternative modelling approaches were then calibrated to the ' \
        'same target data for the 2022 Australian COVID-19 epidemic. ' \
        'The calibration algorithms were also harmonised to the greatest extent possible, ' \
        'although the two analysis approaches that included structure for vaccination ' \
        'required one different parameter to be substituted for the parameters used ' \
        'in the analyses not incorporating this structure. ' \
        'These four approaches were then compared with regards to their fit to ' \
        'the target data, with the analysis that included structure for mobility ' \
        'but not for vaccination found to achieve the highest likelihood. ' \
        f"The `{PRIMARY_ANALYSIS}' analysis was therefore selected as the `main' or `primary' analysis " \
        'and was used for the further analyses, including parameter inference. '
    tex_doc.add_line(description, 'Approach to analyses')
