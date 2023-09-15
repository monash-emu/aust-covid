from typing import List, Union
import arviz as az
from arviz.labels import MapLabeller
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib as mpl
from scipy import stats

from summer2 import CompartmentalModel
from estival.model import BayesianCompartmentalModel
import estival.priors as esp

from emutools.tex import StandardTexDoc
from inputs.constants import SUPPLEMENT_PATH, PLOT_START_DATE, ANALYSIS_END_DATE


def round_sigfig(
    value: float, 
    sig_figs: int
) -> float:
    """
    Round a number to a certain number of significant figures, 
    rather than decimal places.
    
    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    if np.isinf(value):
        return 'infinity'
    else:
        return round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1)) if value != 0.0 else 0.0


def param_table_to_tex(
    param_info: pd.DataFrame,
    prior_names: list,
) -> pd.DataFrame:
    """
    Process aesthetics of the parameter info dataframe into readable information.

    Args:
        param_info: Dataframe with raw parameter information

    Returns:
        table: Ready to write version of the table
    """
    table = param_info[[c for c in param_info.columns if c != 'description']]
    table['value'] = table['value'].apply(lambda x: str(round_sigfig(x, 3) if x != 0.0 else 0.0))  # Round value
    table.loc[[i for i in table.index if i in prior_names], 'values'] = 'Calibrated'  # Suppress value if calibrated
    table.index = param_info['descriptions']  # Use readable description for row names
    table.columns = table.columns.str.replace('_', ' ').str.capitalize()
    table.index.name = None
    table = table[['Value', 'Units', 'Evidence']]  # Reorder columns
    table['Units'] = table['Units'].str.capitalize()
    return table


def get_prior_dist_type(
    prior,
) -> str:
    """
    Clunky way to extract the type of distribution used for a prior.
    
    Args:
        The prior object
    Returns:
        Description of the distribution
    """
    dist_type = str(prior.__class__).replace('>', '').replace("'", '').split('.')[-1].replace('Prior', '')
    return f'{dist_type} distribution'


def get_prior_dist_param_str(
    prior,
) -> str:
    """
    Extract the parameters to the distribution used for a prior.
    Note rounding to three decimal places.
    
    Args:
        prior: The prior objectx

    Returns:
        The parameters to the prior's distribution joined together
    """
    if isinstance(prior, esp.GammaPrior):
        return f'shape: {round(prior.shape, 3)} scale: {round(prior.scale, 3)}'
    else:
        return ' '.join([f'{param}: {round(prior.distri_params[param], 3)}' for param in prior.distri_params])


def get_prior_dist_support(
    prior,
) -> str:
    """
    Extract the bounds to the distribution used for a prior.
    
    Args:
        prior: The prior object

    Returns:        
        The bounds to the prior's distribution joined together
    """
    return ' to '.join([str(round_sigfig(i, 3)) for i in prior.bounds()])


def plot_param_progression(
    idata: az.InferenceData, 
    param_info: pd.DataFrame, 
    request_vars: Union[None, List[str]]=None,
) -> mpl.figure.Figure:
    """
    Plot progression of parameters over model iterations with posterior density plots.
    
    Args:
        idata: Formatted outputs from calibration
        param_info: Collated information on the parameter values (excluding calibration/priors-related)
        request_vars: The parameter names to plot
    
    Returns:
        Formatted figure object created from arviz plotting command
    """
    mpl.rcParams['axes.titlesize'] = 25
    labeller = MapLabeller(var_name_map=param_info['descriptions'])
    trace_plot = az.plot_trace(idata, figsize=(16, 21), compact=False, legend=False, labeller=labeller, var_names=request_vars)
    trace_fig = trace_plot[0, 0].figure
    trace_fig.tight_layout()
    return trace_fig


def plot_posterior_comparison(
    idata: az.InferenceData, 
    priors: list, 
    request_vars: list, 
    display_names: dict,
    dens_interval_req: float,
):
    """
    Area plot posteriors against prior distributions.

    Args:
        idata: Formatted outputs from calibration
        priors: The prior objects
        request_vars: The names of the priors to plot
        display_names: Translation of names to names for display
        dens_interval_req: How much of the central density to plot
    """
    comparison_plot = az.plot_density(
        idata, 
        var_names=request_vars, 
        shade=0.5, 
        labeller=MapLabeller(var_name_map=display_names), 
        point_estimate=None,
        hdi_prob=dens_interval_req,
    );
    req_priors = [p for p in priors if p.name in request_vars]
    for i_ax, ax in enumerate(comparison_plot.ravel()[:len(request_vars)]):
        ax_limits = ax.get_xlim()
        x_vals = np.linspace(ax_limits[0], ax_limits[1], 100)
        y_vals = req_priors[i_ax].pdf(x_vals)
        ax.fill_between(x_vals, y_vals, color='k', alpha=0.2, linewidth=2)
    return comparison_plot[0, 0].figure


def tabulate_priors(
    priors: list, 
    param_info: pd.DataFrame, 
) -> pd.DataFrame:
    """
    Create table of all priors used in calibration algorithm,
    including distribution names, distribution parameters and support.

    Args:
        priors: Priors for use in calibration algorithm
        param_info: Collated information on the parameter values (excluding calibration/priors-related)

    Returns:
        Formatted table combining the information listed above
    """
    names = [param_info['descriptions'][i.name] for i in priors]
    distributions = [get_prior_dist_type(i) for i in priors]
    parameters = [get_prior_dist_param_str(i) for i in priors]
    support = [get_prior_dist_support(i) for i in priors]
    return pd.DataFrame({'Distribution': distributions, 'Parameters': parameters, 'Support': support}, index=names)


def tabulate_param_results(
    idata: az.data.inference_data.InferenceData, 
    priors: list, 
    param_info: pd.DataFrame, 
) -> pd.DataFrame:
    """
    Get tabular outputs from calibration inference object and standardise formatting.

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


def get_negbinom_target_widths(
    targets: pd.Series, 
    idata: az.InferenceData,
    model: CompartmentalModel, 
    base_params: dict, 
    output_name: str, 
    centiles: np.array, 
    prior_names: list,
) -> tuple:
    """
    Get the negative binomial centiles for a given model output 
    and dispersion parameter.

    Args:
        targets: Target time series
        idata: Full inference data
        model: Epidemiological model
        base_params: Default values for all parameters to run through model
        output_name: Name of derived output
        centiles: Centiles to calculate
        prior_names: String names for each priors

    Returns:
        Dataframe with the centiles for the output of interest
        Dispersion parameter used in calculations
    """
    sample_params = az.extract(idata, num_samples=1)
    updated_parameters = base_params | {k: sample_params.variables[k].data[0] for k in prior_names}
    dispersion = sample_params.variables[f'{output_name}_dispersion']
    model.run(parameters=updated_parameters)
    modelled_cases = model.get_derived_outputs_df()[output_name]
    cis = pd.DataFrame(columns=centiles, index=targets.index)
    for time in targets.index:
        mu = modelled_cases.loc[time]
        p = mu / (mu + dispersion)
        cis.loc[time, :] = stats.nbinom.ppf(centiles, dispersion, 1.0 - p)
    return cis, dispersion


def plot_priors(
    priors: list, 
    titles: dict, 
    n_cols: int, 
    n_points: int, 
    rel_overhang: float, 
    prior_cover: float,
) -> go.Figure:
    """
    Plot the PDF of each of a set of priors.

    Args:
        priors: The list of estival prior objects
        titles: Names for the subplots
        n_cols: User request for number of columns
        n_points: Number of points to evaluate the prior at
        rel_overhang: How far out to go past the edge of requested bounds
            (to ensure priors that are discontinuous at their edges go down to zero at the sides)
        prior_cover: How much of the posterior density to cover (before overhanging)

    Returns:
        Multi-panel figure with one panel per prior
    """
    n_cols = 5
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
    fig.update_layout(height=1000, showlegend=False)
    return fig


def plot_spaghetti(
    spaghetti: pd.DataFrame, 
    indicators: List[str], 
    n_cols: int, 
    targets: list,
) -> go.Figure:
    """
    Generate a spaghetti plot to compare any number of requested outputs to targets.

    Args:
        spaghetti: The values from the sampled runs
        indicators: The names of the indicators to look at
        n_cols: Number of columns for the figure
        targets: The calibration targets

    Returns:
        The spaghetti plot figure object
    """
    rows = int(np.ceil(len(indicators) / n_cols))

    fig = make_subplots(rows=rows, cols=n_cols, subplot_titles=indicators)
    for i, ind in enumerate(indicators):
        row = int(np.floor(i / n_cols)) + 1
        col = i % n_cols + 1

        # Model outputs
        ind_spagh = spaghetti[ind]
        ind_spagh.columns = ind_spagh.columns.map(lambda col: f'{col[0]}, {col[1]}')
        ind_spagh = ind_spagh[(PLOT_START_DATE < ind_spagh.index) & (ind_spagh.index < ANALYSIS_END_DATE)]
        fig.add_traces(px.line(ind_spagh).data, rows=row, cols=col)

        # Targets
        target = next((t.data for t in targets if t.name == ind), None)
        if target is not None:
            target = target[(PLOT_START_DATE < target.index) & (target.index < ANALYSIS_END_DATE)]
            target_marker_config = dict(size=15.0, line=dict(width=1.0, color='DarkSlateGrey'))
            lines = go.Scatter(x=target.index, y=target, marker=target_marker_config, name='targets', mode='markers')
            fig.add_trace(lines, row=row, col=col)
    fig.update_layout(showlegend=False, height=400 * rows)
    return fig


def plot_param_hover_spaghetti(
    indicator_spaghetti: pd.DataFrame, 
    idata: az.InferenceData,
) -> go.Figure:
    """
    Generate a spaghetti plot with all parameters displayed on hover.

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
    fig.update_layout(showlegend=False, height=600)
    return fig
