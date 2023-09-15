from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import arviz as az

from summer2 import CompartmentalModel

from aust_covid.inputs import load_household_impacts_data
from aust_covid.tracking import get_param_to_exp_plateau, get_cdr_values
from emutools.tex import StandardTexDoc
from emutools.calibration import get_negbinom_target_widths
from inputs.constants import ANALYSIS_END_DATE, PLOT_START_DATE, SUPPLEMENT_PATH, CHANGE_STR, COLOURS

pd.options.plotting.backend = 'plotly'


def plot_single_run_outputs(model, targets):
    case_targets = next((t.data for t in targets if t.name == 'notifications_ma'))
    death_targets = next((t.data for t in targets if t.name == 'deaths_ma'))
    serosurvey_targets = next((t.data for t in targets if t.name == 'adult_seropos_prop'))

    fig = make_subplots(rows=3, cols=2)
    derived_outputs = model.get_derived_outputs_df()
    x_vals = derived_outputs.index
    fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs['notifications_ma'], name='modelled cases'), row=1, col=1)
    fig.add_trace(go.Scatter(x=case_targets.index, y=case_targets, name='reported cases'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs['deaths_ma'], name='deaths_ma'), row=1, col=2)
    fig.add_trace(go.Scatter(x=death_targets.index, y=death_targets, name='reported deaths ma'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs['adult_seropos_prop'], name='adult seropos'), row=2, col=1)
    fig.add_trace(go.Scatter(x=serosurvey_targets.index, y=serosurvey_targets, name='seropos estimates'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs['reproduction_number'], name='reproduction number'), row=2, col=2)
    for agegroup in model.stratifications['agegroup'].strata:
        fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs[f'deathsXagegroup_{agegroup}'], name=f'{agegroup} deaths'), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs[f'deathsXagegroup_{agegroup}'], name=f'{agegroup} deaths'), row=3, col=2)
    fig['layout']['yaxis6'].update(type='log', range=[-2.0, 2.0])
    fig.update_xaxes(range=(PLOT_START_DATE, ANALYSIS_END_DATE))
    fig.update_layout(height=600, width=1200)
    fig.show()


def plot_subvariant_props(
    spaghetti: pd.DataFrame, 
):
    """
    Plot the proportion of the epidemic attributable to each sub-variant over time.
    Compare against hard-coded key dates of sequencing proportions for each subvariant.

    Args:
        spaghetti: The values from the sampled runs
    """
    fig = go.Figure()

    ba1_results = spaghetti['ba1_prop']
    ba5_results = 1.0 - spaghetti['ba5_prop']
    flattened_cols = [f'chain:{col[0]}, draw:{col[1]}' for col in ba1_results.columns]
    ba1_results.columns = flattened_cols
    ba5_results.columns = flattened_cols
    for c in flattened_cols:
        fig.add_trace(go.Scatter(x=ba1_results.index, y=ba1_results[c]))
        fig.add_trace(go.Scatter(x=ba5_results.index, y=ba5_results[c], line={'dash': 'dot'}))

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

    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=(PLOT_START_DATE, ANALYSIS_END_DATE))
    fig.update_yaxes(range=(0.0, 1.0))
    return fig


def plot_cdr_examples(samples):
    """
    Plot examples of the variation in the case detection rate over time,
    display and include in documentation.

    Args:
        samples: Case detection values
    """
    hh_impact = load_household_impacts_data()
    hh_test_ratio = hh_impact['Proportion testing'] / hh_impact['Proportion symptomatic']
    cdr_values = pd.DataFrame()
    for start_cdr in samples:
        start_cdr = float(start_cdr)
        exp_param = get_param_to_exp_plateau(hh_test_ratio[0], start_cdr)
        cdr_values[round(start_cdr, 3)] = get_cdr_values(exp_param, hh_test_ratio)
    fig = cdr_values.plot(markers=True, labels={'value': 'case detection ratio', 'index': ''})
    return fig


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
    all_targets: list,
    output_colours: dict, 
    req_centiles: np.ndarray, 
    n_samples: int=4, 
    base_alpha: float=0.2, 
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
        all_targets: The selected targets to consider
        output_colours: Colours for plotting the outputs
        req_centiles: Centiles to plot
        n_samples: Number of samples (rows of panels)
        base_alpha: Minimum alpha/transparency for area plots
    """
    fig = go.Figure(layout=go.Layout(width=1000, height=1200))
    targets = [t for t in all_targets if hasattr(t, 'dispersion_param')]
    outputs = [t.name for t in targets]
    fig = make_subplots(rows=n_samples, cols=len(outputs), figure=fig, subplot_titles=[' '] * n_samples * len(outputs))
    up_back_list = get_count_up_back_list(len(req_centiles) - 1)
    alphas = [(a / max(up_back_list)) * (1.0 - base_alpha) + base_alpha for a in up_back_list]
    for i_sample in range(n_samples):
        row = i_sample + 1
        for i_out, o in enumerate(outputs):
            target_extract = targets[i_out].data.loc[PLOT_START_DATE: ANALYSIS_END_DATE]
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
    return fig


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


def plot_processed_mobility(mob_types):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Western Australia', 'weighted average for rest of Australia'])
    fig.update_layout(height=500)
    levels = mob_types['original'].columns
    for p, patch in enumerate(set(levels.get_level_values(0))):
        colour = 0
        for mob_loc in set(levels.get_level_values(1)):
            for m, mob_type in mob_types.items():
                colour += 1
                values = mob_type.loc[:, (patch, mob_loc)]
                fig.add_trace(
                    go.Scatter(x=values.index, y=values, name=f'{m}, {mob_loc}'.replace('_', ' '), showlegend=p==0, line=dict(color=COLOURS[colour])),
                    row=1, col=p + 1,
                )
    return fig


def plot_example_model_matrices(model, parameters):
    epoch = model.get_epoch()
    matrix_func = model.graph.filter('mixing_matrix').get_callable()
    dates = [datetime(2022, month, 1) for month in range(1, 13)]
    agegroups = model.stratifications['agegroup'].strata
    fig = make_subplots(cols=4, rows=3, subplot_titles=[i.strftime('%B') for i in dates])
    fig.update_layout(height=750, width=800)
    for i_date, date in enumerate(dates):
        index = epoch.datetime_to_number(date)
        matrix = matrix_func(model_variables={'time': index}, parameters=parameters)['mixing_matrix']    
        fig.add_trace(
            go.Heatmap(x=agegroups, y=agegroups, z=matrix, zmin=0.0, zmax=6.4), 
            row=int(np.floor(i_date /4) + 1), 
            col=i_date % 4 + 1,
        )
    return fig
