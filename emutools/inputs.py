from pathlib import Path
import pandas as pd
import yaml as yml


def load_param_info(
    data_path: Path, 
) -> pd.DataFrame:
    """
    Load specific parameter information from 
    a ridigly formatted yaml file or crash otherwise.

    Args:
        data_path: Location of the source file
        parameters: The parameters provided by the user (with their values)

    Returns:
        The parameters info DataFrame contains the following fields:
            value: Enough parameter values to ensure model runs, may be over-written in calibration
            descriptions: A brief reader-digestible name/description for the parameter
            units: The unit of measurement for the quantity (empty string if dimensionless)
            evidence: TeX-formatted full description of the evidence underpinning the choice of value
            abbrevaitions: Short name for parameters, e.g. for some plots
            value: The values provided in the parameters argument
    """
    with open(data_path, 'r') as param_file:
        all_data = yml.safe_load(param_file)
    return pd.DataFrame(all_data)
