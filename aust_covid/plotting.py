from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.express.colors import colorbrewer
import arviz as az

from summer2 import CompartmentalModel

from aust_covid.inputs import load_household_impacts_data
from aust_covid.tracking import get_param_to_exp_plateau, get_cdr_values
from aust_covid.mobility import CHANGE_STR
from emutools.tex import StandardTexDoc
from emutools.calibration import melt_spaghetti, get_negbinom_target_widths

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / 'supplement'
COLOURS = colorbrewer.Accent


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
    title_dict = {
        'notifications_ma': 'cases (moving average)',
        'adult_seropos_prop': 'adult seropositive proportion',
        'deaths_ma': 'deaths (moving average)',    
    }
    fig = make_subplots(rows=3, cols=1, subplot_titles=[title_dict[o] for o in outputs])
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


def get_count_up_back_list(
    req_length: int
) -> list:
    """
    Get a list that counts sequentially up from zero and back down again,
    with the total length of the list being that requested.

    Args:
        req_length: Length of requested list

    Returns:
        List containing the sequential integer values
    """
    counting = list(range(req_length))
    count_down = range(round(req_length / 2))[::-1]
    counting[-len(count_down):] = count_down
    return counting


def plot_dispersion_examples(
    idata: az.InferenceData,
    model: CompartmentalModel,
    base_params: list,
    prior_names: list,
    targets: list,
    analysis_start_date: datetime,
    analysis_end_date: datetime,
    output_colours: dict, 
    tex_doc: StandardTexDoc,
    req_centiles: np.ndarray, 
    n_samples: int=4, 
    base_alpha: float=0.2, 
    width: int=1000, 
    height: int=1200,
    show_fig: bool=False,
) -> go.Figure:
    """
    Illustrate the range of the density of the negative
    binomial distribution from some example accepted model
    iterations.

    Args:
        idata: The full inference data
        model: The epidemiological model
        base_params: Default parameter values
        prior_names: The priors used in the calibration algorithm
        targets: The selected targets to consider
        analysis_start_date: Analysis starting time
        analysis_end_date: Analysis end time
        output_colours: Colours for plotting the outputs
        req_centiles: Centiles to plot
        n_samples: Number of samples (rows of panels)
        base_alpha: Minimum alpha/transparency for area plots
        width: Plot width
        height: Plot height
    """
    fig = go.Figure(layout=go.Layout(width=width, height=height))
    outputs = [t.name for t in targets]
    fig = make_subplots(rows=n_samples, cols=len(outputs), figure=fig, subplot_titles=[' '] * n_samples * len(outputs))
    up_back_list = get_count_up_back_list(len(req_centiles) - 1)
    alphas = [(a / max(up_back_list)) * (1.0 - base_alpha) + base_alpha for a in up_back_list]
    for i_sample in range(n_samples):
        row = i_sample + 1
        for i_out, o in enumerate(outputs):
            target_extract = targets[i_out].data.loc[analysis_start_date: analysis_end_date]
            cis, disps = get_negbinom_target_widths(target_extract, idata, model, base_params, o, req_centiles, prior_names)
            col = i_out + 1
            bottom_trace = go.Scatter(x=cis.index, y=cis.iloc[:, 0], line=dict(width=0.0), name='')
            fig.add_traces(bottom_trace, rows=row, cols=col)
            for i_cent, centile in enumerate(cis.columns[1:]):
                colour = f'rgba({output_colours[o]}, {alphas[i_cent]})'
                label = f'{round(cis.columns[i_cent] * 100)} to {round(centile * 100)} centile, {o}'
                mid_trace = go.Scatter(x=cis.index, y=cis[centile], fill='tonexty', line=dict(width=0.0), fillcolor=colour, name=label)
                fig.add_traces(mid_trace, rows=row, cols=col)
            target_trace = go.Scatter(x=target_extract.index, y=target_extract, name=f'reported {o}', mode='markers', marker={'color': f'rgb({output_colours[o]})', 'size': 4})
            fig.add_trace(target_trace, row=row, col=col)
            fig.layout.annotations[i_sample * len(outputs) + i_out].update(text=f'{o}, dispersion param: {round(float(disps.data), 1)}')

    filename = 'dispersion_examples.jpg'
    fig.write_image(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        'Examples of the effect of values of the negative binomial distribution dispersion parameter.', 
        filename,
        'Calibration',
    )
    if show_fig:
        fig.show()


def plot_state_mobility(state_data, jurisdictions, mob_locs):
    fig = make_subplots(rows=4, cols=2, subplot_titles=list(jurisdictions))
    fig.update_layout(height=1500)
    for j, juris in enumerate(jurisdictions):
        for l, mob_loc in enumerate(mob_locs):
            estimates = state_data[state_data['sub_region_1'] == juris][mob_loc]
            legend_str = mob_loc.replace(CHANGE_STR, "").replace("_", " ")
            fig.add_trace(
                go.Scatter(x=estimates.index, y=estimates, name=legend_str, line=dict(color=COLOURS[l]), showlegend=j==0),
                row=j % 4 + 1, col=round(j / 7) + 1,
            )
    return fig


def plot_processed_mobility(model_mob, smoothed_model_mob):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Western Australia', 'weighted average for rest of Australia'])
    fig.update_layout(height=500)
    for p, patch in enumerate(set(model_mob.columns.get_level_values(0))):
        for l, mob_loc in enumerate(set(model_mob.columns.get_level_values(1))):
            values = model_mob.loc[:, (patch, mob_loc)]
            fig.add_trace(
                go.Scatter(x=values.index, y=values, name=mob_loc.replace('_', ' '), line=dict(color=COLOURS[l]), showlegend=p==0),
                row=1, col=p + 1,
            )
            values = smoothed_model_mob.loc[:, (patch, mob_loc)]
            fig.add_trace(
                go.Scatter(x=values.index, y=values, name=f'smoothed_{mob_loc}'.replace('_', ' '), line=dict(color=COLOURS[l + 2]), showlegend=p==0),
                row=1, col=p + 1,
            )
    return fig


def plot_example_model_matrices(model, parameters):
    epoch = model.get_epoch()
    matrix_func = model.graph.filter('mixing_matrix').get_callable()
    dates = [datetime(2022, month, 1) for month in range(1, 13)]
    agegroups = model.stratifications['agegroup'].strata
    fig = make_subplots(cols=4, rows=3, subplot_titles=[i.strftime('%B') for i in dates])
    fig.update_layout(height=700, width=800)
    for i_date, date in enumerate(dates):
        index = epoch.datetime_to_number(date)
        matrix = matrix_func(model_variables={'time': index}, parameters=parameters)['mixing_matrix']    
        fig.add_trace(
            go.Heatmap(x=agegroups, y=agegroups, z=matrix, zmin=0.0, zmax=6.4), 
            row=int(np.floor(i_date /4) + 1), 
            col=i_date % 4 + 1,
        )
    return fig
