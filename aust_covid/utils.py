from typing import Union
from jax import numpy as jnp
from jax import scipy as jsp
import numpy as np
from matplotlib.figure import Figure as MplFig
from plotly.graph_objects import Figure as PlotlyFig

from emutools.tex import StandardTexDoc
from summer2.parameters import Function, Data, DerivedOutput

from inputs.constants import SUPPLEMENT_PATH

def triangle_wave_func(
    time: float, 
    start: float, 
    duration: float, 
    peak: float,
) -> float:
    """
    Generate a peaked triangular wave function
    that starts from and returns to zero.

    Args:
        time: Model time
        start: Time at which wave starts
        duration: Duration of wave
        peak: Peak flow rate for wave
    Returns:
        The wave function
    """
    gradient = peak / (duration * 0.5)
    peak_time = start + duration * 0.5
    time_from_peak = jnp.abs(peak_time - time)
    return jnp.where(time_from_peak < duration * 0.5, peak - time_from_peak * gradient, 0.0)


def convolve_probability(
    source_output: DerivedOutput, 
    density_kernel: Function,
):
    """
    Create function to convolve two processes,
    currently always a modelled derived output and some empirically based distribution.

    Args:
        source_output: Model output over time
        density_kernel: Distribution of delays to the outcome

    Returns:
        Jaxified function to convolve the two processes
    """
    return jnp.convolve(source_output, density_kernel)[:len(source_output)]


def gamma_cdf(
    shape: float, 
    mean: float, 
    x: jnp.array,
) -> jnp.array:
    """
    The regularised gamma function is the CDF of the gamma distribution
    (which is referred to by scipy as "gammainc").

    Args:
        shape: Shape parameter to the desired gamma distribution
        mean: Expectation of the desired gamma distribution
        x: Values to calculate the result over

    Returns:
        Array of CDF values corresponding to input request (x)
    """
    return jsp.special.gammainc(shape, x * shape / mean)


def build_gamma_dens_interval_func(
    shape: float, 
    mean: float, 
    model_times: np.ndarray,
) -> Function:
    """
    Create a function to return the density of the gamma distribution.

    Args:
        shape: Shape parameter to gamma distribution
        mean: Mean of gamma distribution
        model_times: The evaluation times for the model

    Returns:
        Jaxified summer2 function of the distribution
    """
    lags = Data(model_times - model_times[0])
    cdf_values = Function(gamma_cdf, [shape, mean, lags])
    return Function(jnp.gradient, [cdf_values])


def add_image_to_doc(
    fig: Union[MplFig, PlotlyFig], 
    filename: str, 
    caption: str, 
    tex_doc: StandardTexDoc, 
    section: str,
):
    """
    Save an figure image to a local directory and include in TeX doc.

    Args:
        fig: The figure object
        filename: A string for the filenam to save the figure as
        caption: Figure caption for the document
        tex_doc: The working document
        section: Section of the document to include the figure in

    Raises:
        TypeError: If the figure is not one of the two supported formats
    """
    full_filename = f'{filename}.jpg'
    if isinstance(fig, MplFig):
        fig.savefig(SUPPLEMENT_PATH / full_filename)
    elif isinstance(fig, PlotlyFig):
        fig.write_image(SUPPLEMENT_PATH / full_filename)
    else:
        raise TypeError('Figure type not supported')
    tex_doc.include_figure(caption, full_filename, section)
