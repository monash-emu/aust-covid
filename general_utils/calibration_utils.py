import arviz as az
from arviz.labels import MapLabeller
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
from scipy import stats

from summer2 import CompartmentalModel
from estival.model import BayesianCompartmentalModel
import estival.priors as esp

from general_utils.tex_utils import StandardTexDoc

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / 'supplement'


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
    table = param_info.iloc[:, 1:]  # Drop description for now
    table['value'] = table['value'].apply(lambda x: str(round_sigfig(x, 3) if x != 0.0 else 0.0))  # Round value
    table.loc[[i for i in table.index if i in prior_names], 'value'] = 'Calibrated'  # Suppress value if calibrated
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
        prior: The prior object

    Returns:
        The parameters to the prior's distribution joined together
    """
    if isinstance(prior, esp.GammaPrior):
        return f'shape: {round(prior.shape, 3)} scale: {round(prior.scale, 3)}'
    elif isinstance(prior, esp.BetaPrior):
        return ' '.join([f'{param}: {round(prior.distri_params[param][0], 3)}' for param in prior.distri_params])
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
    tex_doc: StandardTexDoc,
    show_fig: bool=False,
    request_vars=None,
    name_ext: str='',
) -> mpl.figure.Figure:
    """
    Plot progression of parameters over model iterations with posterior density plots.
    
    Args:
        idata: Formatted outputs from calibration
        param_info: Collated information on the parameter values (excluding calibration/priors-related)
    
    Returns:
        Formatted figure object created from arviz plotting command
    """
    mpl.rcParams['axes.titlesize'] = 25
    trace_plot = az.plot_trace(
        idata, 
        figsize=(16, 21), 
        compact=False, 
        legend=False,
        labeller=MapLabeller(var_name_map=param_info['descriptions']),
        var_names=request_vars,
    )
    trace_fig = trace_plot[0, 0].figure
    trace_fig.tight_layout()

    filename = f'traces{name_ext}.jpg'
    trace_fig.savefig(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        'Parameter posteriors and traces by chain.', 
        filename,
        'Calibration', 
    )
    if show_fig:
        trace_fig.show()


def plot_param_posterior(
    idata: az.data.inference_data.InferenceData, 
    display_names: dict, 
    tex_doc: StandardTexDoc,
    show_fig: bool=False,
    request_vars=None,
    name_ext: str='',
) -> mpl.figure.Figure:
    """
    Plot posterior distribution of parameters.

    Args:
        idata: Formatted outputs from calibration
        param_info: Collated information on the parameter values (excluding calibration/priors-related)
        tex_doc: 
            
    Returns:
        Formatted figure object created from arviz plotting command
    """
    plot = az.plot_posterior(
        idata,
        figsize=(16, 21), 
        labeller=MapLabeller(var_name_map=display_names),
        var_names=request_vars,
    )
    fig = plot[0, 0].figure;
    
    filename = f'posteriors{name_ext}.jpg'
    fig.savefig(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        'Parameter posteriors, chains combined.', 
        filename,
        'Calibration', 
    )

    if show_fig:
        fig.show()


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


def sample_idata(
    idata: az.InferenceData, 
    n_samples: int, 
    model: BayesianCompartmentalModel,
) -> pd.DataFrame:
    """
    Sample from inference data, retain only data pertaining to the parameters,
    and sort by draw and chain.

    Args:
        idata: The inference data
        n_samples: Number of samples to select
        model: The Bayesian calibration object, only used for getting parameter names

    Returns:
        Sampled data converted to dataframe with columns for parameters and multi-index for chain and draw
    """
    prior_names = [k for k in model.priors.keys() if 'dispersion' not in k]
    return az.extract(idata, num_samples=n_samples).to_dataframe()[prior_names].sort_index(level='draw').sort_index(level='chain')


def get_sampled_outputs(
    model: BayesianCompartmentalModel, 
    sampled_idata: pd.DataFrame, 
    outputs: list, 
    parameters: dict,
) -> pd.DataFrame:
    """
    Take output of sample_idata and run through model to get results of multiple runs of model.

    Args:
        model: The Bayesian calibration object, needed to run the parameters through to get results again
        sampled_idata: Output of sampled_idata
        outputs: The names of the derived outputs we want to examine
        parameters: The base parameter names to be updated by the samples

    Returns:
        Sampled runs with multi-index for columns output, chain and draw, and index being independent variable of model (i.e. time)
    """
    spaghetti_df = pd.concat([pd.DataFrame(columns=sampled_idata.index)] * len(outputs), keys=outputs, axis=1)
    for (chain, draw), params in sampled_idata.iterrows():
        run_results = model.run(parameters | params.to_dict()).derived_outputs
        for output in outputs:
            spaghetti_df[(output, chain, draw)] = run_results[output]
    return spaghetti_df


def melt_spaghetti(
    spaghetti_df: pd.DataFrame, 
    output: str, 
    sampled_idata: az.InferenceData,
) -> pd.DataFrame:
    """
    Take output of get_sampled_outputs, 'melt'/convert to long format and add columns for parameter values.
    This is done in preparation for producing a 'spaghetti plot' of sequential model runs.

    Args:
        spaghetti_df: Output of get_sampled_outputs
        output: The name of the derived output of interest
        idata: The output of sample_idata

    Returns:
        The melted sequential runs spaghetti
    """
    melted_df = spaghetti_df[output].melt(ignore_index=False)
    for (chain, draw), params in sampled_idata.iterrows():
        for param, value in params.iteritems():
            melted_df.loc[(melted_df['chain']==chain) & (melted_df['draw'] == draw), param] = round_sigfig(value, 3)
    return melted_df


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
