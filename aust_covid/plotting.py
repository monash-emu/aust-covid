from datetime import datetime, timedelta
from typing import List, Dict, Set
import pandas as pd
pd.options.plotting.backend = 'plotly'
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import arviz as az
from xarray.core.variable import Variable

from summer2 import CompartmentalModel

from aust_covid.inputs import load_national_case_data, load_owid_case_data, load_case_targets, load_who_death_data, load_serosurvey_data
from inputs.constants import PLOT_START_DATE, ANALYSIS_END_DATE, AGE_STRATA
from emutools.tex import DummyTexDoc
from aust_covid.inputs import load_household_impacts_data, get_subvariant_prop_dates
from aust_covid.tracking import get_param_to_exp_plateau, get_cdr_values
from emutools.calibration import get_negbinom_target_widths, get_target_from_name
from emutools.plotting import get_row_col_for_subplots
from inputs.constants import ANALYSIS_END_DATE, PLOT_START_DATE, CHANGE_STR, COLOURS, RUN_IDS


def get_n_rows_plotly_fig(
    fig: go.Figure,
) -> int:
    """Get the number of rows in a plotly figure,
    whether or not multi-panel.

    Args:
        fig: The figure to consider

    Returns:
        The number of rows
    """
    try:
        rows = fig._get_subplot_rows_columns()[0].stop - 1
    except:
        rows = 1
    return rows


def format_output_figure(
    fig: go.Figure, 
) -> go.Figure:
    """Standard formatting for a figure of a model output
    or outputs over time (multi-panel or otherwise).

    Args:
        fig: The figure

    Returns:
        The adjusted figure
    """
    rows = get_n_rows_plotly_fig(fig)
    heights = [320, 600, 900, 900]
    fig.update_xaxes(range=(PLOT_START_DATE, ANALYSIS_END_DATE))
    return fig.update_layout(height=heights[rows - 1])


def get_count_up_back_list(
    req_length: int
) -> list:
    """Get a list that counts sequentially up from zero and back down again,
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


def plot_single_run_outputs(
    model: CompartmentalModel, 
    targets: list,
) -> go.Figure:
    """Produce standard plot of selected model outputs.

    Args:
        model: The summer epidemiological model
        targets: All the targets being used for calibration/optimisation

    Returns:
        The figure
    """
    panels = [
        'cases',
        'deaths',
        'seropositive',
        'reproduction number',
        'daily deaths by age',
        'daily deaths by age (log scale)',
    ]
    fig = make_subplots(rows=3, cols=2, subplot_titles=panels)
    derived_outputs = model.get_derived_outputs_df()
    x_vals = derived_outputs.index
    output_map = {
        'notifications_ma': [1, 1],
        'deaths_ma': [1, 2],
        'adult_seropos_prop': [2, 1],
        'reproduction_number': [2, 2],
    }
    for out in output_map:
        fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs[out], name=f'modelled {out}'.replace('_', ' ')), row=output_map[out][0], col=output_map[out][1])
        if out != 'reproduction_number':
            target = get_target_from_name(targets, out)
            fig.add_trace(go.Scatter(x=target.index, y=target, name=f'target {out}'.replace('_', ' ')), row=output_map[out][0], col=output_map[out][1])
    for agegroup in model.stratifications['agegroup'].strata:
        for col in range(1, 3):
            fig.add_trace(go.Scatter(x=x_vals, y=derived_outputs[f'deathsXagegroup_{agegroup}'], name=f'{agegroup} deaths'), row=3, col=col)
    fig['layout']['yaxis6'].update(type='log', range=[-2.0, 2.0])
    fig.update_xaxes(tickangle=45)
    return format_output_figure(fig)


def plot_subvariant_props(
    spaghetti: pd.DataFrame, 
) -> go.Figure:
    """Plot the proportion of the epidemic attributable to each sub-variant over time.
    Compare against hard-coded key dates of sequencing proportions for each subvariant.

    Args:
        spaghetti: The values from the sampled runs
    
    Returns:
        The figure
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
    voc_emerge_df = get_subvariant_prop_dates()
    lag = timedelta(days=3.5)  # Because dates are given as first day of week in which VoC was first detected
    for voc in voc_emerge_df:
        voc_info = voc_emerge_df[voc]
        colour = voc_info['colour']
        fig.add_vline(voc_info['any'] + lag, line_dash='dot', line_color=colour)
        fig.add_vline(voc_info['>1%'] + lag, line_dash='dash', line_color=colour)
        fig.add_vline(voc_info['>50%'] + lag, line_color=colour)
    fig.update_yaxes(range=(0.0, 1.0))
    fig.update_layout(showlegend=False)
    return format_output_figure(fig)


def plot_cdr_examples(
    samples: Variable,
) -> go.Figure:
    """Plot examples of the variation in the case detection rate over time.

    Args:
        samples: Case detection values

    Returns:
        The figure
    """
    hh_impact = load_household_impacts_data()
    hh_test_ratio = hh_impact['testing'] / hh_impact['symptomatic']
    cdr_values = pd.DataFrame()
    for start_cdr in samples:
        start_cdr = float(start_cdr)
        exp_param = get_param_to_exp_plateau(hh_test_ratio[0], start_cdr)
        cdr_values[round(start_cdr, 3)] = get_cdr_values(exp_param, hh_test_ratio)
    fig = cdr_values.plot(markers=True, labels={'value': 'case detection ratio', 'index': ''}).update_layout(legend_title='starting value')
    return format_output_figure(fig)


def plot_dispersion_examples(
    idata: az.InferenceData,
    model: CompartmentalModel,
    base_params: Dict[str, float],
    prior_names: List[str],
    all_targets: list,
    output_colours: Dict[str, str], 
    req_centiles: np.ndarray, 
    n_samples: int=3, 
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

    Returns:
        The figure
    """
    targets = [t for t in all_targets if hasattr(t, 'dispersion_param')]
    outputs = [t.name for t in targets]
    fig = make_subplots(rows=n_samples, cols=len(outputs), subplot_titles=[' '] * n_samples * len(outputs), vertical_spacing=0.13)
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
            out = o.replace('_ma', '')
            target_trace = go.Scatter(x=target_extract.index, y=target_extract, name=f'reported {out}', mode='markers', marker={'color': f'rgb({output_colours[o]})', 'size': 4})
            fig.add_trace(target_trace, row=row, col=col)
            fig.layout.annotations[i_sample * len(outputs) + i_out].update(text=f'{out}, dispersion param: {round(float(disps.data), 1)}')
    fig = format_output_figure(fig)
    return fig.update_xaxes(tickangle=45)


def plot_state_mobility(
    state_data: pd.DataFrame, 
    jurisdictions: Set[str], 
    mob_locs: List[str],
) -> go.Figure:
    """Plot the raw Australian Google mobility data by state/jurisdiction.

    Args:
        state_data: Google mobility data for jurisdictions of Australia
        jurisdictions: The names of the states/jurisdictions
        mob_locs: Names identifying the locations for mobility measurement

    Returns:
        The figure
    """
    fig = make_subplots(rows=4, cols=2, subplot_titles=list(jurisdictions), vertical_spacing=0.06)
    for j, juris in enumerate(jurisdictions):
        for l, mob_loc in enumerate(mob_locs):
            estimates = state_data[state_data['sub_region_1'] == juris][mob_loc]
            legend_str = mob_loc.replace(CHANGE_STR, '').replace('_', ' ')
            trace = go.Scatter(x=estimates.index, y=estimates, name=legend_str, line=dict(color=COLOURS[l]))
            fig.add_trace(trace, row=j % 4 + 1, col=round(j / 7) + 1)
    fig.update_yaxes(range=(-90, 70))
    return format_output_figure(fig)


def plot_processed_mobility(
    mobility_types: Dict[str, pd.DataFrame]
)-> go.Figure:
    """Illustrate the process of converting from raw data 
    to mobility functions fed into the model.

    Args:
        mobility_types: Dataframes for the data at each stage of processing

    Returns:
        The output figure
    """
    patch_map = {
        'wa': 'Western Australia',
        'non wa': 'rest of Australia',
    }
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(patch_map.values()))
    for m, mob_type in enumerate(mobility_types):
        mob_data = mobility_types[mob_type]
        for p, patch in enumerate(set(mob_data.columns.get_level_values(0))):
            for l, mob_loc in enumerate(set(mob_data.columns.get_level_values(1))):
                values = mob_data.loc[:, (patch, mob_loc)]
                trace_name = f'{mob_loc}, {patch_map[patch]}, {mob_type}'
                mob_trace = go.Scatter(x=values.index, y=values, name=trace_name, line=dict(color=COLOURS[m + l * 3]))
                fig.add_trace(mob_trace, row=1, col=p + 1)
    return format_output_figure(fig)


def plot_example_model_matrices(
    model: CompartmentalModel,
    parameters: Dict[str, float],
) -> go.Figure:
    """Plot example mixing matrices at each month in 2022.

    Args:
        model: The epidemiological model
        parameters: Parameter values (presumably optimised for likelihood)

    Returns:
        The figure
    """
    model.finalize()
    epoch = model.get_epoch()
    matrix_func = model.graph.filter('mixing_matrix').get_callable()
    dates = [datetime(2022, month, 1) for month in range(1, 13)]
    agegroups = model.stratifications['agegroup'].strata
    n_cols = 4
    fig = make_subplots(cols=n_cols, rows=3, subplot_titles=[i.strftime('%B') for i in dates], vertical_spacing=0.1)
    for i_date, date in enumerate(dates):
        index = epoch.datetime_to_number(date)
        matrix = matrix_func(model_variables={'time': index}, parameters=parameters)['mixing_matrix']
        heatmap = go.Heatmap(x=agegroups, y=agegroups, z=matrix, zmin=0.0, zmax=6.4)
        row, col = get_row_col_for_subplots(i_date, n_cols)
        fig.add_trace(heatmap, row=row, col=col)
    return format_output_figure(fig)


def plot_full_vacc(
    full_vacc_masks: List[str], 
    df: pd.DataFrame,
    prop_df: pd.DataFrame,
) -> go.Figure:
    """Plot full (two) dose vaccination coverage by age group
    over time as absolute number and proportion.

    Args:
        full_vacc_masks: Strings identifying the needed columns
        df: The vaccination dataframe
        prop_df: The adjusted dataframe with proportions instead of numbers (from get_full_vacc_props)

    Returns:
        The figure
    """
    fig = make_subplots(2, 1, vertical_spacing=0.1, subplot_titles=['number', 'proportion'])
    for a, age in enumerate(full_vacc_masks):
        prop_age = int(np.round(a / len(full_vacc_masks) * 250.0))
        colour = f'rgb({prop_age},{250 - prop_age},250)'
        trace_name = age.replace('- Number of people fully vaccinated', '').replace('Age group - ', '')
        data = df[age].dropna()
        prop_data = prop_df[age].dropna()
        fig.add_trace(go.Scatter(x=data.index, y=data, name=trace_name, line={'color': colour}), row=1, col=1)
        fig.add_trace(go.Scatter(x=prop_data.index, y=prop_data, name=trace_name, line={'color': colour}), row=2, col=1)
    return fig.update_layout(height=600, legend_title='age group')


def plot_program_coverage(
    program_masks: List[str], 
    df: pd.DataFrame,
) -> go.Figure:
    """
    Plot vaccination coverage by program across four panels to represent the main programs.

    Args:
        program_masks: Strings identifying the needed columns
        df: The vaccination dataframe

    Returns:
        The plotly figure object
    """
    fig = make_subplots(rows=4, cols=1, subplot_titles=list(program_masks.keys()), vertical_spacing=0.12)
    for m, mask in enumerate(program_masks):
        fig.add_traces(px.line(df[program_masks[mask]]).data, rows=m + 1, cols=1)
    return fig.update_layout(height=600, showlegend=False)


def plot_immune_props(
    model: CompartmentalModel,
    vacc_df: pd.DataFrame,
    lag_vacc_df: pd.DataFrame,
) -> go.Figure:
    epoch = model.get_epoch()
    age_breaks = ['5', '15']
    titles = ['Modelled 5 to 9 age group', 'Modelled 15 and above age groups']
    fig = make_subplots(2, 1, subplot_titles=titles, vertical_spacing=0.1)
    for i_plot, age in enumerate(age_breaks):
        cols = [f'prop_{age}_{imm}' for imm in model.stratifications['immunity'].strata][::-1]
        model_vacc_df = model.get_derived_outputs_df()[cols]
        model_vacc_df.columns = model_vacc_df.columns.str.replace('_', ' ')
        fig.add_traces(model_vacc_df.plot.area().data, i_plot + 1, 1)
    dfs = {'raw': vacc_df, 'lagged': lag_vacc_df}
    for data_type in dfs:
        for i, pop in enumerate(['primary full', 'adult booster']):
            pop_str = 'National - Population 5-11' if pop == 'primary full' else 'National - Population 16 and over'
            line_type = 'dash' if data_type == 'raw' else 'dot'
            line_style = {'color': 'black', 'dash': line_type}
            x_vals = dfs[data_type].index
            y_vals = dfs[data_type][pop] / dfs[data_type][pop_str]
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, line=line_style, name='coverage'), row=i + 1, col=1)
    fig.update_xaxes(range=epoch.index_to_dti([model.times[0], model.times[-1]]))
    fig.update_yaxes(range=[0.0, 1.0])
    return fig.update_layout(height=600)


def plot_targets(targets, for_plotly: bool=True):
    dummy_doc = DummyTexDoc()
    subplot_specs = [
        [{'colspan': 2}, None], 
        [{}, {}],
    ]
    fig = make_subplots(rows=2, cols=2, specs=subplot_specs)
    combined_data = load_case_targets(dummy_doc)
    national_data = load_national_case_data(dummy_doc)
    serosurvey_targets = get_target_from_name(targets, 'adult_seropos_prop')
    if for_plotly:
        fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data, name='combined cases'), row=1, col=1)
        fig.add_trace(go.Scatter(x=serosurvey_targets.index, y=serosurvey_targets, name='serosurvey target'), row=2, col=2)
    thin_line = {'width': 0.8}
    fig.add_trace(go.Scatter(x=national_data.index, y=national_data, name='national cases', line=thin_line), row=1, col=1)
    owid_data = load_owid_case_data(dummy_doc)
    fig.add_trace(go.Scatter(x=owid_data.index, y=owid_data, name='owid cases', line=thin_line), row=1, col=1)
    case_targets = get_target_from_name(targets, 'notifications_ma')
    fig.add_trace(go.Scatter(x=case_targets.index, y=case_targets, name='final case target (smoothed)'), row=1, col=1)
    thin_line = {'width': 1.2}
    death_data = load_who_death_data(dummy_doc)
    fig.add_trace(go.Scatter(x=death_data.index, y=death_data, name='who deaths', line=thin_line), row=2, col=1)
    death_targets = get_target_from_name(targets, 'deaths_ma')
    fig.add_trace(go.Scatter(x=death_targets.index, y=death_targets, name='death target (smoothed)', line=thin_line), row=2, col=1)
    serosurvey_data = load_serosurvey_data(dummy_doc)
    fig.add_trace(go.Scatter(x=serosurvey_data.index, y=serosurvey_data, name='serosurvey data'), row=2, col=2)
    serosurvey_ceiling = get_target_from_name(targets, 'seropos_ceiling')
    fig.add_trace(go.Scatter(x=serosurvey_ceiling.index, y=serosurvey_ceiling, name='seroprevalence ceiling'), row=2, col=2)
    fig.update_xaxes(range=(PLOT_START_DATE, ANALYSIS_END_DATE))
    return fig.update_layout(height=420)


def plot_multi_spaghetti(
    spaghettis: Dict[str, pd.DataFrame], 
    output: str, 
    targets: list,
):
    target = next(i for i in targets if i.name == output)
    n_cols = 2
    fig = make_subplots(rows=2, cols=n_cols, subplot_titles=list(RUN_IDS.keys()), shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.08)
    for i, analysis in enumerate(RUN_IDS.keys()):
        row, col = get_row_col_for_subplots(i, n_cols)
        spaghetti = spaghettis[analysis][output]
        spaghetti.columns = [f'{str(chain)}, {str(draw)}' for chain, draw in spaghetti.columns]    
        fig.add_traces(spaghetti.plot().data, rows=row, cols=col)
        fig.add_trace(go.Scatter(x=target.data.index, y=target.data, mode='markers', marker={'color': 'black', 'size': 12}), row=row, col=col)
    fig.update_xaxes(range=(PLOT_START_DATE, ANALYSIS_END_DATE))
    return fig.update_layout(height=600, margin={i: 30 for i in ['t', 'b', 'l', 'r']})


def plot_vacc_implementation(
    df: pd.DataFrame,
) -> go.Figure:
    """Illustrate the process of calculating between stratum transitions.

    Args:
        df: Augmented vaccination data

    Returns:
        Plot in three vertical panels
    """
    fig = make_subplots(3, 1, subplot_titles=['persons vaccinated', 'coverage', 'rates implemented'], vertical_spacing=0.08)
    fig.add_traces(df[['primary full', 'adult booster']].plot().data, rows=1, cols=1)
    fig.add_traces(df[['prop primary full', 'prop adult booster']].plot().data, rows=2, cols=1)
    fig.add_traces(df[['rate primary full', 'rate adult booster']].plot().data, rows=3, cols=1)
    return fig.update_layout(height=800, showlegend=False, margin={'t': 40})


def plot_mixing_matrices(
    matrices: Dict[str, np.array],
) -> go.Figure:
    """Plot mixing matrices from their standard location-specific format.

    Args:
        matrices: The matrices

    Returns:
        Interactive figure
    """
    n_cols = 2
    fig = make_subplots(2, n_cols, vertical_spacing=0.08, horizontal_spacing=0.07, subplot_titles=[k.replace('_', ' ') for k in matrices])
    for i, matrix in enumerate(matrices):
        row, col = get_row_col_for_subplots(i, n_cols)
        fig.add_traces(px.imshow(matrices[matrix], x=AGE_STRATA, y=AGE_STRATA).data, rows=row, cols=col)
    return fig.update_layout(height=800, width=850, margin={'t': 40})


def plot_3d_spaghetti(
    indicator_name: str, 
    spaghetti: pd.DataFrame, 
    targets: list, 
    target_freq=5,
) -> go.Figure:
    """Create interactive plotly figure for comparing outputs against targets.

    Args:
        indicator_name: Name of indicator to consider
        spaghetti: The outputs from the sequential calibration runs
        targets: The targets from the calibration algorithm
        target_freq: How frequently to intersperse target plots between sequential runs

    Returns:
        The interactive figure
    """
    fig = go.Figure()
    sample = spaghetti.loc[spaghetti.index > PLOT_START_DATE, indicator_name]
    sample.columns = sample.columns.map(lambda x: ', '.join([*map(str, x)]))
    target = get_target_from_name(targets, indicator_name)
    for i_col, col in enumerate(sample.columns):
        ypos = [i_col] * len(sample.index)
        fig.add_trace(go.Scatter3d(x=sample.index, y=ypos, z=sample[col], name=col, mode='lines', line={'width': 5.0}))
        if target is not None and i_col % target_freq == 0:
            target = target[target.index < ANALYSIS_END_DATE]
            fig.add_trace(go.Scatter3d(x=target.index, y=ypos, z=target, name='target', mode='markers', marker={'size': 1.0}, line={'color': 'black'}))
    fig.update_yaxes(showticklabels=False)
    return fig.update_layout(height=800, scene=dict(xaxis_title='', yaxis_title='run', zaxis_title=indicator_name, yaxis={'showticklabels': False}))


def plot_matrices_3d(
    matrices: Dict[str, np.array],
) -> go.Figure:
    """Plot interactive 3D surface plots of matrices.

    Args:
        matrices: The mixing matrices by location

    Returns:
        The 4-panel plot
    """
    fig_type = {'type': 'surface'}
    n_cols = 2
    fig = make_subplots(
        rows=2, 
        cols=2, 
        specs=[[fig_type, fig_type], [fig_type, fig_type]], 
        horizontal_spacing=0.02, 
        vertical_spacing=0.05,
        subplot_titles=[k.replace('_', ' ') for k in matrices.keys()],
    )
    for l, location in enumerate(matrices):
        row, col = get_row_col_for_subplots(l, n_cols)
        fig.add_trace(go.Surface(x=AGE_STRATA, y=AGE_STRATA, z=matrices[location], showscale=False), row=row, col=col)
    scene_req = {'zaxis': {'range': (0.0, 3.0), 'title': 'contacts', 'dtick': 1.0}, 'xaxis': {'title': ''}, 'yaxis': {'title': ''}}
    return fig.update_layout(
        scene1=scene_req, scene2=scene_req, scene3=scene_req, scene4=scene_req, 
        height=800, 
        margin={'t': 25, 'b': 25, 'l': 25, 'r': 25},
    )
