from jax import numpy as jnp
from jax import scipy as jsp
import numpy as np
import pandas as pd
import yaml as yml

from summer2.parameters import Function, Data, DerivedOutput

from aust_covid.constants import DATA_PATH


def round_sigfig(value: float, sig_figs: int) -> float:
    """
    Round a number to a certain number of significant figures,
    rather than decimal places.

    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    if np.isinf(value):
        return "infinity"
    else:
        return (
            round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1)) if value != 0.0 else 0.0
        )


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
    return jnp.convolve(source_output, density_kernel)[: len(source_output)]


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
    with open(DATA_PATH / "parameters.yml", "r") as param_file:
        param_info = yml.safe_load(param_file)

    # Check each loaded set of keys (parameter names) are the same as the arbitrarily chosen first key
    first_key_set = param_info[list(param_info.keys())[0]].keys()
    for cat in param_info:
        working_keys = param_info[cat].keys()
        if working_keys != first_key_set:
            msg = f"Keys to {cat} category: {working_keys} - do not match first category {first_key_set}"
            raise ValueError(msg)

    return pd.DataFrame(param_info)


def param_table_to_tex(
    param_info: pd.DataFrame,
    prior_names: list,
) -> pd.DataFrame:
    """Process aesthetics of the parameter info dataframe into
    readable information that can be output to TeX.

    Args:
        param_info: Dataframe with raw parameter information

    Returns:
        table: Ready to write version of the table
    """
    table = param_info[[c for c in param_info.columns if c != "description"]]
    table["value"] = table["value"].apply(
        lambda x: str(round_sigfig(x, 3) if x != 0.0 else 0.0)
    )  # Round
    table.loc[[i for i in table.index if i in prior_names], "value"] = (
        "Calibrated"  # Suppress value if calibrated
    )
    table.index = param_info["descriptions"]  # Readable description for row names
    table.columns = table.columns.str.replace("_", " ").str.capitalize()
    table.index.name = None
    table = table[["Value", "Units", "Evidence"]]  # Reorder columns
    table["Units"] = table["Units"].str.capitalize()
    return table


def get_target_from_name(
    targets: list,
    name: str,
) -> pd.Series:
    """Get the data for a specific target from a set of targets from its name.

    Args:
        targets: All the targets
        name: The name of the desired target

    Returns:
        Single target to identify
    """
    return next((t.data for t in targets if t.name == name), None)
