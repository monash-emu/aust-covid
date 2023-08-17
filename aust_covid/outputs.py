from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from aust_covid.inputs import load_household_impacts_data
from aust_covid.model import get_param_to_exp_plateau, get_cdr_values
from general_utils.tex_utils import StandardTexDoc
from general_utils.calibration_utils import melt_spaghetti

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / 'supplement'


def plot_key_outputs(
    sampled_idata: pd.DataFrame, 
    output_results: pd.DataFrame, 
    start_date: datetime, 
    end_date: datetime,
    tex_doc: StandardTexDoc, 
    outputs: list,
    case_targets: pd.Series,
    serosurvey_targets: pd.Series,
    death_targets: pd.Series,
    show_fig: bool=False,
):
    """
    Create spaghetti plot of key outputs over sequential model runs.

    Args:
        sampled_idata: Sample from the inference data
        output_results: Outputs that have been run through the model
        start_date: Start date for plot
        end_date: End date for plot
        tex_doc: TeX documentation object
        outputs: Names of outputs to plot
        case_targets: Notification series for comparison
        serosurvey_targets: Serosurvey values for comparison
        death_targets: Death series for comparison
        show_fig: Whether to display the figure now
    """
    fig = make_subplots(rows=3, cols=1, subplot_titles=outputs)
    for i_out, out in enumerate(outputs):
        spaghetti = melt_spaghetti(output_results, out, sampled_idata)
        lines = px.line(spaghetti, y='value', color='chain', line_group='draw', hover_data=spaghetti.columns)
        fig.add_traces(lines.data, rows=i_out + 1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=case_targets.index, 
            y=case_targets, 
            name='reported cases',
            mode='markers',
            marker={'color': 'LightBlue', 'size': 4, 'line': {'color': 'black', 'width': 1}},
        ), 
        row=1, 
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=serosurvey_targets.index, 
            y=serosurvey_targets, 
            name='serosurveys',
            mode='markers',
            marker={'color': 'white', 'size': 20, 'line': {'color': 'black', 'width': 1}},
        ), 
        row=2, 
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=death_targets.index, 
            y=death_targets, 
            name='reported deaths ma', 
            mode='markers',
            marker={'color': 'Red', 'size': 4, 'line': {'color': 'black', 'width': 1}},
        ),
        row=3, 
        col=1,
    )
    fig.update_xaxes(range=(start_date, end_date))
    fig.update_layout(height=1000, width=1000)

    filename = 'key_outputs.jpg'
    fig.write_image(SUPPLEMENT_PATH / filename)
    caption = 'Key results for randomly sampled runs from calibration algorithm.'
    tex_doc.include_figure(
        caption, 
        filename,
        'Results',
    )
    if show_fig:
        fig.show()


def plot_subvariant_props(
    sampled_idata: pd.DataFrame, 
    output_results: pd.DataFrame, 
    start_date: datetime, 
    end_date: datetime,
    tex_doc: StandardTexDoc, 
    show_fig: bool=False,
):
    """
    Plot the proportion of the epidemic attributable to each sub-variant over time.
    Compare against hard-coded key dates of sequencing proportions for each subvariant.

    Args:
        sampled_idata: Sample from the inference data
        output_results: Outputs that have been run through the model
        start_date: Start date for plot
        end_date: End date for plot
        tex_doc: TeX documentation object
        show_fig: Whether to display the figure now
    """
    fig = go.Figure()
    ba1_results = melt_spaghetti(output_results, 'ba1_prop', sampled_idata)
    ba5_results = melt_spaghetti(output_results, 'ba5_prop', sampled_idata)
    ba5_results['value'] = 1.0 - ba5_results['value']  # Flip BA.5 results
    fig.add_traces(px.line(ba1_results, y='value', color='chain', line_group='draw', hover_data=ba1_results.columns).data)
    fig.add_traces(px.line(ba5_results, y='value', color='chain', line_group='draw', hover_data=ba5_results.columns).data)
    voc_emerge_df = pd.DataFrame(
        {
            'ba1': [datetime(2021, 11, 22), datetime(2021, 11, 29), datetime(2021, 12, 20), 'blue'],
            'ba2': [datetime(2021, 11, 29), datetime(2022, 1, 10), datetime(2022, 3, 7), 'red'], 
            'ba5': [datetime(2022, 3, 28), datetime(2022, 5, 16), datetime(2022, 6, 27), 'green'],
        },
        index=['any', '>1%', '>50%', 'colour']
    )
    lag = timedelta(days=3.5)  # Dates are given as first day of week in which VoC was first detected
    for voc in voc_emerge_df:
        voc_info = voc_emerge_df[voc]
        colour = voc_info['colour']
        fig.add_vline(voc_info['any'] + lag, line_dash='dot', line_color=colour)
        fig.add_vline(voc_info['>1%'] + lag, line_dash='dash', line_color=colour)
        fig.add_vline(voc_info['>50%'] + lag, line_color=colour)
    fig.update_xaxes(range=(start_date, end_date))
    fig.update_yaxes(range=(0.0, 1.0))

    filename = 'subvariant_props.jpg'
    fig.write_image(SUPPLEMENT_PATH / filename)
    caption = 'Proportion of modelled cases attributable to each subvariant over time. ' \
        'Key dates for each variant shown as vertical bars: blue, BA.1; red, BA.2; green, BA.5; ' \
        'dotted, first detection; dashed, \>1\%; solid, \>50\%. '
    tex_doc.include_figure(
        caption, 
        filename,
        'Results',
    )
    if show_fig:
        fig.show()


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
