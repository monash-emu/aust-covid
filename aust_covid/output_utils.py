import pandas as pd
import arviz as az
import pymc as pm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def convert_idata_to_df(
    idata: az.data.inference_data.InferenceData, 
    param_names: list,
) -> pd.DataFrame:
    """
    Convert arviz inference data to dataframe organised
    by draw and chain through multi-indexing.
    
    Args:
        idata: arviz inference data
        param_names: String names of the model parameters
    """
    sampled_idata_df = idata.to_dataframe()[param_names]
    return sampled_idata_df.sort_index(level="draw").sort_index(level="chain")


def run_samples_through_model(
    samples_df: pd.DataFrame, 
    model: pm.model.Model,
) -> pd.DataFrame:
    """
    Run parameters dataframe in format created by convert_idata_to_df
    through epidemiological model to get outputs of interest.
    
    Args:
        samples_df: Parameters to run through in format generated from convert_idata_to_df
        model: Model to run them through
    """
    sres = pd.DataFrame(index=model.model._get_ref_idx(), columns=samples_df.index)
    for (chain, draw), params in samples_df.iterrows():
        sres[(chain,draw)] = model.run(params.to_dict()).derived_outputs["notifications"]
    return sres


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


def plot_from_model_runs_df(
    model_results: pd.DataFrame, 
    sampled_df: pd.DataFrame,
    param_names: list,
) -> go.Figure:
    """
    Create interactive plot of model outputs by draw and chain
    from standard data structures.
    
    Args:
        model_results: Model outputs generated from run_samples_through_model
        sampled_df: Inference data converted to dataframe in output format of convert_idata_to_df
    """
    melted = model_results.melt(ignore_index=False)
    melted.columns = ["chain", "draw", "notifications"]

    # Add parameter values from sampled dataframe to plotting 
    for (chain, draw), params in sampled_df.iterrows():
        for p in param_names:
            melted.loc[(melted["chain"]==chain) & (melted["draw"] == draw), p] = round_sigfig(params[p], 3)
        
    return px.line(melted, y="notifications", color="chain", line_group="draw", hover_data=melted.columns)
