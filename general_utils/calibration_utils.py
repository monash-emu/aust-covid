from pylatex.utils import NoEscape
import arviz as az
from arviz.labels import MapLabeller
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib as mpl

from estival.model import BayesianCompartmentalModel

from general_utils.tex_utils import StandardTexDoc

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"


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
    return round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1))


def param_table_to_tex(
    param_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process parameter info dataframe into the actual parameter information

    Args:
        param_info: Dataframe with parameter information

    Returns:
        Version of param_info ready to write into LaTeX document
    """
    param_table = param_info.iloc[:, 1:]
    param_table.index = param_info['descriptions']
    param_table.columns = param_table.columns.str.replace('_', ' ').str.capitalize()
    param_table.index.name = None
    param_table['Manual values'] = param_table['Manual values'].apply(lambda x: str(round_sigfig(x, 3) if x != 0.0 else 0.0))
    return param_table


def get_fixed_param_value_text(
    param: str,
    parameters: dict,
    param_units: dict,
    prior_names: list,
    decimal_places=2,
    calibrated_string="Calibrated, see priors table",
) -> str:
    """
    Get the value of a parameter being used in the model for the parameters table,
    except indicate that it is calibrated if it's one of the calibration parameters.
    
    Args:
        param: Parameter name
        parameters: All parameters expected by the model
        param_units: The units for the parameter being considered
        prior_names: The names of the parameters used in calibration
        decimal_places: How many places to round the value to
        calibrated_string: The text to use if the parameter is calibrated

    Returns:
        Description of the parameter value
    """
    return calibrated_string if param in prior_names else f"{round(parameters[param], decimal_places)} {param_units[param]}"


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
    dist_type = str(prior.__class__).replace(">", "").replace("'", "").split(".")[-1].replace("Prior", "")
    return f"{dist_type} distribution"


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
    return " ".join([f"{param}: {round(prior.distri_params[param], 3)}" for param in prior.distri_params])


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
    return " to ".join([str(round_sigfig(i, 3)) for i in prior.bounds()])


def plot_param_progression(
    idata: az.data.inference_data.InferenceData, 
    param_info: pd.DataFrame, 
    tex_doc: StandardTexDoc,
    show_fig: bool=False,
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
        figsize=(16, 3 * len(idata.posterior)), 
        compact=False, 
        legend=True,
        labeller=MapLabeller(var_name_map=param_info['descriptions']),
    )
    trace_fig = trace_plot[0, 0].figure
    trace_fig.tight_layout()

    filename = 'traces.jpg'
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
    param_info: pd.DataFrame, 
    tex_doc: StandardTexDoc,
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
    posterior_plot = az.plot_posterior(
        idata,
        labeller=MapLabeller(var_name_map=param_info['descriptions']),
    )
    posterior_fig = posterior_plot[0, 0].figure;
    
    filename = 'posteriors.jpg'
    posterior_fig.savefig(SUPPLEMENT_PATH / filename)
    tex_doc.include_figure(
        'Parameter posteriors, chains combined.', 
        filename,
        'Calibration', 
    )


def tabulate_parameters(
    parameters: dict, 
    priors: list, 
    param_info: pd.DataFrame, 
) -> pd.DataFrame:
    """
    Create table of all parameters being consumed by model,
    with the values being used and evidence to support them.

    Args:
        parameters: All parameter values, even if calibrated
        priors: Priors for use in calibration algorithm
        param_info: Collated information on the parameter values (excluding calibration/priors-related)

    Returns:
        Formatted table combining the information listed above
    """
    values_column = [get_fixed_param_value_text(i, parameters, param_info["units"], priors) for i in parameters]
    evidence_column = [NoEscape(param_info["evidence"][i]) for i in parameters]
    names_column = [param_info["descriptions"][i] for i in parameters]
    return pd.DataFrame({"Value": values_column, "Evidence": evidence_column}, index=names_column)


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
    names = [param_info["descriptions"][i.name] for i in priors]
    distributions = [get_prior_dist_type(i) for i in priors]
    parameters = [get_prior_dist_param_str(i) for i in priors]
    support = [get_prior_dist_support(i) for i in priors]
    return pd.DataFrame({"Distribution": distributions, "Parameters": parameters, "Support": support}, index=names)


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
    results_table = az.summary(idata)
    results_table.index = [param_info["descriptions"][p.name] for p in priors]
    for col_to_round in ["mean", "hdi_3%", "hdi_97%"]:
        results_table[col_to_round] = results_table.apply(lambda x: str(round_sigfig(x[col_to_round], 3)), axis=1)
    results_table["hdi"] = results_table.apply(lambda x: f"{x['hdi_3%']} to {x['hdi_97%']}", axis=1)    
    results_table = results_table.drop(["mcse_mean", "mcse_sd", "hdi_3%", "hdi_97%"], axis=1)
    results_table.columns = ["Mean", "Standard deviation", "ESS bulk", "ESS tail", "R_hat", "High-density interval"]
    return results_table


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
    return az.extract(idata, num_samples=n_samples).to_dataframe()[model.priors.keys()].sort_index(level='draw').sort_index(level='chain')


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
