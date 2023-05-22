from pylatex.utils import NoEscape
import arviz as az
from arviz.labels import MapLabeller
from pathlib import Path
import pandas as pd
import datetime
import plotly.graph_objects as go
import matplotlib as mpl

from estival.model import BayesianCompartmentalModel

from aust_covid.output_utils import convert_idata_to_df, run_samples_through_model, plot_from_model_runs_df, round_sigfig

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"


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

    Return:
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
    Return:
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
    Return:
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
    Return:        
        The bounds to the prior's distribution joined together
    """
    return " to ".join([str(i) for i in prior.bounds()])


def graph_param_progression(
    idata: az.data.inference_data.InferenceData, 
    descriptions: dict, 
):
    """
    Plot progression of parameters over model iterations with posterior density plots.
    
    Args:
        uncertainty_outputs: Formatted outputs from calibration
        descriptions: Parameter descriptions
    """
    mpl.rcParams["axes.titlesize"] = 25
    trace_plot = az.plot_trace(
        idata, 
        figsize=(16, 3 * len(idata.posterior)), 
        compact=False, 
        legend=True,
        labeller=MapLabeller(var_name_map=descriptions),
    )
    trace_fig = trace_plot[0, 0].figure
    trace_fig.tight_layout()
    return trace_fig


def graph_param_posterior(
    idata: az.data.inference_data.InferenceData, 
    descriptions: dict, 
    grid_request: tuple=None,
):
    """
    Plot posterior distribution of parameters.

    Args:
        uncertainty_outputs: Formatted outputs from calibration
        descriptions: Parameter descriptions
        grid_request: How the subplots should be arranged
    """
    posterior_plot = az.plot_posterior(
        idata,
        labeller=MapLabeller(var_name_map=descriptions),
        grid=grid_request,
    )
    posterior_plot = posterior_plot[0, 0].figure
    return posterior_plot


def graph_sampled_outputs(
    idata: az.data.inference_data.InferenceData, 
    n_samples: int, 
    output: str, 
    bayesian_model: BayesianCompartmentalModel, 
    target_data: pd.Series,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
):
    """
    Plot sample model runs from the calibration algorithm.

    Args:
        uncertainty_outputs: Outputs from calibration
        n_samples: Number of times to sample from calibration data
        output: The output of interest
        bayesian_model: The calibration model (that contains the epi model, priors and targets)
        target_data: Comparison data to plot against
    """
    prior_names = bayesian_model.priors.keys()
    sampled_idata = az.extract(idata, num_samples=n_samples)  # Sample from the inference data
    sampled_df = convert_idata_to_df(sampled_idata, prior_names)
    sample_model_results = run_samples_through_model(sampled_df, bayesian_model, output)  # Run through epi model
    fig = plot_from_model_runs_df(sample_model_results, sampled_df, prior_names, start_date, end_date)
    fig.add_trace(go.Scatter(x=target_data.index, y=target_data, marker=dict(color="black"), name=output, mode="markers"))
    return fig


def tabulate_parameters(parameters, param_units, priors, param_descriptions, param_evidence):
    values_column = [get_fixed_param_value_text(i, parameters, param_units, priors) for i in parameters]
    evidence_column = [NoEscape(param_evidence[i]) for i in parameters]
    names_column = [param_descriptions[i] for i in parameters]
    return pd.DataFrame({"Value": values_column, "Evidence": evidence_column}, index=names_column)


def tabulate_priors(priors, descriptions):

    names = [descriptions[i.name] for i in priors]
    distributions = [get_prior_dist_type(i) for i in priors]
    parameters = [get_prior_dist_param_str(i) for i in priors]
    support = [get_prior_dist_support(i) for i in priors]
    return pd.DataFrame({"Distribution": distributions, "Parameters": parameters, "Support": support}, index=names)


def tabulate_param_results(
    idata: az.data.inference_data.InferenceData, 
    priors: list, 
    param_descriptions: dict,
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
    results_table.index = [param_descriptions[p.name] for p in priors]
    for col_to_round in ["mean", "hdi_3%", "hdi_97%"]:
        results_table[col_to_round] = results_table.apply(lambda x: str(round_sigfig(x[col_to_round], 3)), axis=1)
    results_table["hdi"] = results_table.apply(lambda x: f"{x['hdi_3%']} to {x['hdi_97%']}", axis=1)    
    results_table = results_table.drop(["mcse_mean", "mcse_sd", "hdi_3%", "hdi_97%"], axis=1)
    results_table.columns = ["Mean", "Standard deviation", "ESS bulk", "ESS tail", "R_hat", "High-density interval"]
    return results_table
