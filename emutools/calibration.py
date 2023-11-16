from typing import List, Dict, Optional
import arviz as az
from arviz.labels import MapLabeller
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
pd.options.plotting.backend = 'plotly'

import estival.priors as esp

from inputs.constants import PLOT_START_DATE, ANALYSIS_END_DATE, RUN_IDS, RUNS_PATH, BURN_IN
from emutools.plotting import get_row_col_for_subplots
from emutools.utils import get_target_from_name, round_sigfig

from aust_covid.plotting import get_standard_subplot_fig


def get_prior_dist_type(prior) -> str:
    """Find the type of distribution used for a prior.
    
    Args:
        The prior

    Returns:
        Description of the distribution
    """
    dist_type = str(prior.__class__).replace('>', '').replace("'", '').split('.')[-1].replace('Prior', '')
    return f'{dist_type} distribution'


def get_prior_dist_param_str(prior) -> str:
    """Extract the parameters to the distribution used for a prior,
    rounding to three decimal places.
    
    Args:
        prior: The prior

    Returns:
        The parameters to the prior's distribution joined together
    """
    if isinstance(prior, esp.GammaPrior):
        return f'shape: {round(prior.shape, 3)} scale: {round(prior.scale, 3)}'
    else:
        return ' '.join([f'{param}: {round(prior.distri_params[param], 3)}' for param in prior.distri_params])


def get_prior_dist_support(prior) -> str:
    """Extract the bounds to the distribution used for a prior.
    
    Args:
        prior: The prior

    Returns:        
        The bounds to the prior's distribution joined together
    """
    return ' to '.join([str(round_sigfig(i, 3)) for i in prior.bounds()])


def plot_param_progression(
    idata: az.InferenceData, 
    descriptions: pd.Series, 
    req_vars: Optional[List[str]]=None,
) -> mpl.figure.Figure:
    """Plot progression of parameters over model iterations with posterior density plots.
    
    Args:
        idata: Formatted outputs from calibration
        descriptions: Short parameter names
        req_vars: The parameter names to plot
    
    Returns:
        The figure
    """
    labeller = MapLabeller(var_name_map=descriptions)
    trace_plot = az.plot_trace(idata, figsize=(15, 21), compact=False, legend=False, labeller=labeller, var_names=req_vars)
    trace_fig = trace_plot[0, 0].figure
    trace_fig.tight_layout()
    plt.close()
    return trace_fig


def plot_posterior_comparison(
    idata: az.InferenceData, 
    priors: list, 
    req_vars: list, 
    display_names: dict,
    span: float,
) -> plt.Figure:
    """Area plot posteriors against prior distributions.

    Args:
        idata: Formatted outputs from calibration
        priors: The prior objects
        req_vars: The names of the priors to plot
        display_names: Translation of names to names for display
        span: How much of the central density to plot
    
    Returns:
        The figure
    """
    labeller = MapLabeller(var_name_map=display_names)
    comparison_plot = az.plot_density(idata, var_names=req_vars, shade=0.5, labeller=labeller, point_estimate=None, hdi_prob=span, grid=grid)
    req_priors = [p for p in priors if p.name in req_vars]
    for i_ax, ax in enumerate(comparison_plot.ravel()[:len(req_vars)]):
        ax_limits = ax.get_xlim()
        x_vals = np.linspace(ax_limits[0], ax_limits[1], 100)
        y_vals = req_priors[i_ax].pdf(x_vals)
        ax.fill_between(x_vals, y_vals, color='k', alpha=0.2, linewidth=2)
    plt.close()
    return comparison_plot[0, 0].figure


def tabulate_priors(
    priors: list, 
    param_info: pd.DataFrame, 
) -> pd.DataFrame:
    """Create table of all priors used in calibration algorithm,
    including distribution names, distribution parameters and support.

    Args:
        priors: Priors for use in calibration algorithm
        param_info: Collated information on the parameter values (excluding calibration/priors-related)

    Returns:
        Formatted table explaining the priors used
    """
    names = [param_info['descriptions'][i.name] for i in priors]
    distributions = [get_prior_dist_type(i) for i in priors]
    parameters = [get_prior_dist_param_str(i) for i in priors]
    support = [get_prior_dist_support(i) for i in priors]
    return pd.DataFrame({'Distribution': distributions, 'Parameters': parameters, 'Support': support}, index=names)


def tabulate_calib_results(
    idata: az.data.inference_data.InferenceData, 
    priors: list, 
    param_info: pd.DataFrame, 
) -> pd.DataFrame:
    """
    Get tabular outputs from calibration inference object, 
    except for the dispersion parameters, and standardise formatting.

    Args:
        uncertainty_outputs: Outputs from calibration
        priors: Model priors
        param_descriptions: Short names for parameters used in model

    Returns:
        Calibration results table in standard format
    """
    table = az.summary(idata)
    table = table[~table.index.str.contains('_dispersion')]
    table.index = [param_info['descriptions'][p.name] for p in priors]
    for col_to_round in ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk', 'ess_tail', 'r_hat']:
        table[col_to_round] = table.apply(lambda x: str(round_sigfig(x[col_to_round], 3)), axis=1)
    table['hdi'] = table.apply(lambda x: f'{x["hdi_3%"]} to {x["hdi_97%"]}', axis=1)    
    table = table.drop(['mcse_mean', 'mcse_sd', 'hdi_3%', 'hdi_97%'], axis=1)
    table.columns = ['Mean', 'Standard deviation', 'ESS bulk', 'ESS tail', '\\textit{\^{R}}', 'High-density interval']
    return table


def plot_priors(
    priors: list, 
    titles: pd.Series, 
    n_cols: int, 
    n_points: int, 
    rel_overhang: float, 
    prior_cover: float,
) -> go.Figure:
    """Plot the PDF of each of a set of priors.

    Args:
        priors: The list of estival prior objects
        titles: Names for the subplots
        n_cols: User request for number of columns
        n_points: Number of points to evaluate the prior at
        rel_overhang: How far out to go past the edge of requested bounds
            (to ensure priors that are discontinuous at their edges go down to zero at the sides)
        prior_cover: How much of the posterior density to cover (before overhanging)

    Returns:
        The figure
    """
    n_rows = int(np.ceil(len(priors) / n_cols))
    titles = [titles[p.name] for p in priors]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)
    for p, prior in enumerate(priors):
        extremes = prior.ppf(1.0 - prior_cover), prior.ppf(prior_cover)
        overhang = (extremes[1] - extremes[0]) * rel_overhang
        x_values = np.linspace(extremes[0] - overhang, extremes[1] + overhang, n_points)
        y_values = [prior.pdf(x) for x in x_values]
        row = int(np.floor(p / n_cols)) + 1
        col = p % n_cols + 1
        fig.add_trace(go.Scatter(x=x_values, y=y_values, fill='tozeroy'), row=row, col=col)
    return fig.update_layout(height=800, showlegend=False)


def plot_spaghetti(
    spaghetti: pd.DataFrame, 
    indicators: List[str], 
    n_cols: int, 
    targets: list,
) -> go.Figure:
    """Generate a spaghetti plot to compare any number of requested outputs to targets.

    Args:
        spaghetti: The values from the sampled runs
        indicators: The names of the indicators to look at
        n_cols: Number of columns for the figure
        targets: The calibration targets

    Returns:
        The spaghetti plot figure object
    """
    rows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(rows, n_cols, [i.replace('_', ' ') for i in indicators])
    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)

        # Model outputs
        ind_spagh = spaghetti[ind]
        ind_spagh.columns = ind_spagh.columns.map(lambda col: f'{col[0]}, {col[1]}')
        fig.add_traces(px.line(ind_spagh).data, rows=row, cols=col)

        # Targets
        if ind != 'reproduction_number':
            target = get_target_from_name(targets, ind)
            target_marker_config = dict(size=15.0, line=dict(width=1.0, color='DarkSlateGrey'))
            lines = go.Scatter(x=target.index, y=target, marker=target_marker_config, name='targets', mode='markers')
            fig.add_trace(lines, row=row, col=col)
            fig.update_layout(yaxis4=dict(range=[0, 2.2]))
    return fig.update_xaxes(range=[PLOT_START_DATE, ANALYSIS_END_DATE])


def plot_param_hover_spaghetti(
    indicator_spaghetti: pd.DataFrame, 
    idata: az.InferenceData,
) -> go.Figure:
    """Generate a spaghetti plot with all parameters displayed on hover.

    Args:
        indicator_spaghetti: The values from the sampled runs for one indicator only
        idata: The corresponding inference data

    Returns:
        The spaghetti plot figure object
    """
    fig = go.Figure()
    working_data = pd.DataFrame()
    for col in indicator_spaghetti.columns:
        chain, draw = col
        working_data['values'] = indicator_spaghetti[col]
        info = {i: float(j) for i, j in dict(idata.posterior.sel(chain=int(chain), draw=int(draw)).variables).items()}
        for param in info:
            working_data[param] = int(info[param]) if param in ['chain', 'draw'] else round_sigfig(info[param], 3)
        lines = px.line(working_data, y='values', hover_data=working_data.columns)
        fig.add_traces(lines.data)
    fig.update_xaxes(range=[PLOT_START_DATE, ANALYSIS_END_DATE])
    return fig.update_layout(showlegend=False, height=500)


def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame], 
    targets: list, 
    outputs: List[str], 
    analysis: str, 
    quantiles: List[float], 
    max_alpha: float=0.7
) -> go.Figure:
    """Plot the credible intervals with subplots for each output,
    for a single run of interest.

    Args:
        quantile_outputs: Dataframes containing derived outputs of interest for each analysis type
        targets: Calibration targets
        output: User-requested output of interest
        analysis: The key for the analysis type
        quantiles: User-requested quantiles for the patches to be plotted over
        max_alpha: Maximum alpha value to use in patches

    Returns:
        The interactive figure
    """
    n_cols = 2
    target_names = [t.name for t in targets]
    fig = make_subplots(rows=2, cols=n_cols, subplot_titles=[o.replace('_ma', '').replace('_', ' ') for o in outputs], vertical_spacing=0.1, horizontal_spacing=0.04)
    analysis_data = quantile_outputs[analysis]
    for i, output in enumerate(outputs):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = analysis_data[output]
        for q, quant in enumerate(quantiles):
            alpha = min((q, len(quantiles) - q)) / np.floor(len(quantiles) / 2) * max_alpha
            fill_colour = f'rgba(0,30,180,{str(alpha)})'
            fig.add_traces(go.Scatter(x=data.index, y=data[quant], fill='tonexty', fillcolor=fill_colour, line={'width': 0}, name=quant), rows=row, cols=col)
        fig.add_traces(go.Scatter(x=data.index, y=data[0.5], line={'color': 'black'}, name='median'), rows=row, cols=col)
        if output in target_names:
            target = get_target_from_name(targets, output)
            marker_format = {'size': 10.0, 'color': 'rgba(250, 135, 206, 0.2)', 'line': {'width': 1.0}}
            fig.add_traces(go.Scatter(x=target.index, y=target, mode='markers', marker=marker_format, name=target.name), rows=row, cols=col)
    fig.update_xaxes(range=[PLOT_START_DATE, ANALYSIS_END_DATE])
    return fig.update_layout(yaxis4={'range': [0.0, 2.5]}, height=800, showlegend=False, margin={'t': 40, 'b': 40, 'l': 40, 'r': 40})


def plot_output_ranges_by_analysis(
    quantile_outputs: Dict[str, pd.DataFrame], 
    targets: list, 
    output: str, 
    quantiles: List[float], 
    max_alpha: float=0.7
) -> go.Figure:
    """Plot the credible intervals with subplots for each analysis type,
    for a single output of interest.

    Args:
        quantile_outputs: Dataframes containing derived outputs of interest for each analysis type
        targets: Calibration targets
        output: User-requested output of interest
        quantiles: User-requested quantiles for the patches to be plotted over
        max_alpha: Maximum alpha value to use in patches

    Returns:
        The interactive figure
    """
    n_cols = 2
    target_names = [t.name for t in targets]
    fig = make_subplots(rows=2, cols=n_cols, subplot_titles=list(RUN_IDS.keys()), shared_yaxes=True, vertical_spacing=0.1)
    for a, analysis in enumerate(RUN_IDS):
        row, col = get_row_col_for_subplots(a, n_cols)
        analysis_data = quantile_outputs[analysis]
        data = analysis_data[output]
        for q, quant in enumerate(quantiles):
            alpha = min((q, len(quantiles) - q)) / np.floor(len(quantiles) / 2) * max_alpha
            fill_colour = f'rgba(0,30,180,{str(alpha)})'
            fig.add_traces(go.Scatter(x=data.index, y=data[quant], fill='tonexty', fillcolor=fill_colour, line={'width': 0}, name=quant), rows=row, cols=col)
        fig.add_traces(go.Scatter(x=data.index, y=data[0.5], line={'color': 'black'}, name='median'), rows=row, cols=col)
        if output in target_names:
            target = get_target_from_name(targets, output)
            marker_format = {'size': 10.0, 'color': 'rgba(250, 135, 206, 0.2)', 'line': {'width': 1.0}}
            fig.add_traces(go.Scatter(x=target.index, y=target, mode='markers', marker=marker_format, name=target.name), rows=row, cols=col)
    fig.update_xaxes(range=[PLOT_START_DATE, ANALYSIS_END_DATE])
    return fig.update_layout(height=500, showlegend=False)


def get_like_components(
    components: List[str]
) -> List[pd.DataFrame]:
    """Get dictionary containing one dataframe 
    for each requested contribution to the likelihood,
    with columns for each analysis type and integer index.
    
    Args:
        User requested likelihood components
    """
    like_outputs = {}
    for comp in components:
        like_outputs[comp] = pd.DataFrame(columns=list(RUN_IDS.keys()))
        for analysis, run_id in RUN_IDS.items():
            working_data = pd.read_hdf(RUNS_PATH / run_id / 'output/results.hdf', 'likelihood')[comp]
            like_outputs[comp][analysis] = working_data
    return like_outputs


def plot_like_components_by_analysis(
    like_outputs: Dict[str, pd.DataFrame], 
    plot_type: str, 
    clips: Dict[str, tuple]={},
    alpha: float=0.2,
    linewidth: float=1.0,
) -> plt.figure:
    """Use seaborn plotting functions to show likelihood components from various runs.

    Args:
        like_outputs: Output from get_like_components above
        plot_type: Type of seaborn plot
        clips: Lower clips for the components' x-axis range

    Returns:
        The analysis comparison figure
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
    axes = axes.reshape(-1)
    plotter = getattr(sns, plot_type)
    legend_plot_types = ['kdeplot', 'histplot']
    title_map = {
        'loglikelihood': 'total likelihood',
        'll_adult_seropos_prop': 'seroprevalence contribution',
        'll_deaths_ma': 'deaths contribution',
        'll_notifications_ma': 'cases contribution',
    }
    for m, comp in enumerate(like_outputs.keys()):
        clip = clips[comp] if clips else None
        kwargs = {'common_norm': False, 'clip': clip, 'fill': True, 'alpha': alpha, 'linewidth': linewidth} if plot_type == 'kdeplot' else {}        
        ax = axes[m]
        plotter(like_outputs[comp].loc[:, BURN_IN:, :], ax=ax, **kwargs)
        ax.set_title(title_map[comp])
        if m == 0 and plot_type in legend_plot_types:
            sns.move_legend(ax, loc='upper left')
        elif plot_type in legend_plot_types:
            ax.legend_.set_visible(False)
    fig.tight_layout()
    return fig


def get_outcome_df_by_chain() -> Dict[str, pd.DataFrame]:
    """Compile dictionary of dataframes for each analysis type,
    each with column multi-index for likelihood component
    and chain number.

    Returns:
        Compiled data structure
    """
    like_dfs = {}
    for analysis, run_id in RUN_IDS.items():
        like_df = pd.read_hdf(RUNS_PATH / run_id / 'output/results.hdf', 'likelihood')
        like_df['chain'] = like_df.index.get_level_values(0)
        like_df['index'] = like_df.index.get_level_values(1)
        like_dfs[analysis] = like_df.pivot(index='index', columns=['chain'])
    return like_dfs
