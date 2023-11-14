from jax import numpy as jnp
from jax import scipy as jsp
import numpy as np
import pandas as pd
import yaml as yml

from summer2.parameters import Function, Data, DerivedOutput

from inputs.constants import INPUTS_PATH


def triangle_wave_func(
    time: float, 
    start: float, 
    duration: float, 
    peak: float,
) -> float:
    """Generate a peaked triangular wave function
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
) -> jnp.array:
    """Create function to convolve two processes,
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
    """The regularised gamma function is the CDF of the gamma distribution
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
    """Create a function to return the density of the gamma distribution.

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


def capture_kwargs(*args, **kwargs):
    return kwargs


def load_param_info() -> pd.DataFrame:
    """
    Load specific parameter information from a ridigly formatted yaml file, and crash otherwise.

    Returns:
        The parameters info DataFrame contains the following fields:
            value: Enough parameter values to ensure model runs, may be over-written in calibration
            descriptions: A brief reader-digestible name/description for the parameter
            units: The unit of measurement for the quantity (empty string if dimensionless)
            evidence: TeX-formatted full description of the evidence underpinning the choice of value
            abbreviations: Short names for parameters, e.g. for some plots
    """
    with open(INPUTS_PATH / 'parameters.yml', 'r') as param_file:
        param_info = yml.safe_load(param_file)

    # Check each loaded set of keys (parameter names) are the same as the arbitrarily chosen first key
    first_key_set = param_info[list(param_info.keys())[0]].keys()
    for cat in param_info:
        working_keys = param_info[cat].keys()
        if working_keys != first_key_set:
            msg = f'Keys to {cat} category: {working_keys} - do not match first category {first_key_set}'
            raise ValueError(msg)
    
    return pd.DataFrame(param_info)
