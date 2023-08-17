import pandas as pd

from aust_covid.inputs import load_household_impacts_data
from aust_covid.model import get_param_to_exp_plateau, get_cdr_values


def plot_cdr_examples(samples):
    hh_impact = load_household_impacts_data()
    hh_test_ratio = hh_impact['Proportion testing'] / hh_impact['Proportion symptomatic']
    cdr_values = pd.DataFrame()
    for start_cdr in samples:
        exp_param = get_param_to_exp_plateau(hh_test_ratio[0], start_cdr)
        cdr_values[round(start_cdr, 3)] = get_cdr_values(exp_param, hh_test_ratio)
    fig = cdr_values.plot(markers=True, labels={'value': 'case detection ratio', 'index': ''})
    fig.show()
