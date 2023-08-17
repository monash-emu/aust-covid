from pathlib import Path
import pandas as pd

from aust_covid.inputs import load_household_impacts_data
from aust_covid.model import get_param_to_exp_plateau, get_cdr_values
from general_utils.tex_utils import StandardTexDoc

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / 'supplement'


def plot_cdr_examples(
    samples: pd.Series, 
    tex_doc: StandardTexDoc, 
    show_fig: bool=False,
):
    """
    Plot examples of the variation in the case detection rate over time,
    display and include in documentation.

    Args:
        samples: Case detection values
        tex_doc: TeX documentation object
        show_fig: Whether to display the figure now
    """
    hh_impact = load_household_impacts_data()
    hh_test_ratio = hh_impact['Proportion testing'] / hh_impact['Proportion symptomatic']
    cdr_values = pd.DataFrame()
    for start_cdr in samples:
        exp_param = get_param_to_exp_plateau(hh_test_ratio[0], start_cdr)
        cdr_values[round(start_cdr, 3)] = get_cdr_values(exp_param, hh_test_ratio)
    fig = cdr_values.plot(markers=True, labels={'value': 'case detection ratio', 'index': ''})

    filename = 'cdr_samples.jpg'
    fig.write_image(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        'Examples of simulated case detection rates over modelled time.', 
        filename,
        'Outputs',
        subsection='Notifications', 
    )
    if show_fig:
        fig.show()
