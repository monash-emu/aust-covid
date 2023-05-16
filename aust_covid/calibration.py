import pylatex as pl
from pylatex.utils import NoEscape
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

from estival.model import BayesianCompartmentalModel

from aust_covid.doc_utils import DocumentedProcess, FigElement, TextElement, TableElement
from aust_covid.model import build_aust_model
from aust_covid.output_utils import convert_idata_to_df, run_samples_through_model, round_sigfig, plot_from_model_runs_df

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


class DocumentedCalibration(DocumentedProcess):
    def __init__(
        self, 
        model: BayesianCompartmentalModel,
        outputs: az.data.inference_data.InferenceData,
        priors: list, 
        parameters: dict,
        descriptions: dict, 
        units: dict, 
        evidence: dict, 
        start: datetime,
        end: datetime,
        doc: pl.Document=None,
    ):
        """
        Supports calibration of a summer model,
        with documentation to TeX document as the processes proceed.
        Most of this should be general enough to use for any summer calibration.

        Args:
            priors: The prior objects
            targets: The targets to fit to
            iterations: The number of iterations to run
            burn_in: The number of iterations to discard as burn-in
            model_func: The function to build the model
            parameters: The base parameter requests before updating through calibration
            descriptions: Strings to describe the parameters properly
            units: Strings for the units of each parameter
            evidence: Strings with a more detailed description of the evidence for each parameter
            start: Starting date for simulation
            end: Finish date for simulation
            doc: The TeX document to populate
        """
        super().__init__(doc, True)
        self.bayesian_model = model
        self.uncertainty_outputs = outputs
        self.priors = priors
        self.prior_names = [priors[i_prior].name for i_prior in range(len(priors))]
        self.params = parameters
        self.descriptions = descriptions
        self.units = units
        self.evidence = evidence
        self.model = build_aust_model(start, end, None, add_documentation=False)
       
    def graph_param_progression(self):
        """
        Plot progression of parameters over model iterations with posterior density plots.
        """
        trace_plot = az.plot_trace(self.uncertainty_outputs, figsize=(16, 3.0 * len(self.uncertainty_outputs.posterior)), compact=False, legend=True)
        for i_prior, prior in enumerate(self.priors):
            for i_col, column in enumerate(["posterior", "trace"]):
                ax = trace_plot[i_prior][i_col]
                ax.set_title(f"{self.descriptions[prior.name]}, {column}", fontsize=20)
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.set_tick_params(labelsize=15)

        location = "progression.jpg"
        plt.savefig(SUPPLEMENT_PATH / location)
        self.add_element_to_doc("Calibration", FigElement(location))

    def graph_param_posterior(self):
        """
        Plot posterior distribution of parameters.
        """
        az.plot_posterior(data=self.uncertainty_outputs)
        location = "posterior.jpg"
        plt.savefig(SUPPLEMENT_PATH / location)
        self.add_element_to_doc("Calibration", FigElement(location))

    def graph_sampled_outputs(self, n_samples, output):
        sampled_idata = az.extract(self.uncertainty_outputs, num_samples=n_samples)  # Sample from the inference data
        sampled_df = convert_idata_to_df(sampled_idata, self.prior_names)
        sample_model_results = run_samples_through_model(sampled_df, self.bayesian_model, output)
        data = self.bayesian_model.targets[output].data
        fig = plot_from_model_runs_df(sample_model_results, sampled_df, self.prior_names)
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data, 
                marker=dict(color="black"), 
                name=output, 
                mode="markers",
            ),
        )
        filename = "calibration_fit.jpg"
        fig.write_image(SUPPLEMENT_PATH / filename)
        self.add_element_to_doc("Calibration", FigElement(filename))

    def add_calib_table_to_doc(self):
        """
        Report calibration input choices in table.
        """
        text = "Input parameters varied through calibration with uncertainty distribution parameters and support. \n"
        self.add_element_to_doc("Calibration", TextElement(text))

        headers = ["Name", "Distribution", "Distribution parameters", "Support"]
        col_widths = "p{2.7cm} " * 4
        rows = []
        for prior in self.priors:
            prior_desc = self.descriptions[prior.name]
            dist_type = get_prior_dist_type(prior)
            dist_params = get_prior_dist_param_str(prior)
            dist_range = get_prior_dist_support(prior)
            rows.append([prior_desc, dist_type, dist_params, dist_range])
        self.add_element_to_doc("Calibration", TableElement(col_widths, headers, rows))

    def table_param_results(self):
        """
        Report results of calibration analysis.
        """
        calib_summary = az.summary(self.uncertainty_outputs)
        headers = ["Para-meter", "Mean (SD)", "3-97% high-density interval", "MCSE mean (SD)", "ESS bulk", "ESS tail", "R_hat"]
        rows = []
        for param in calib_summary.index:
            summary_row = calib_summary.loc[param]
            name = self.descriptions[param]
            mean_sd = f"{summary_row['mean']} ({summary_row['sd']})"
            hdi = f"{summary_row['hdi_3%']} to {summary_row['hdi_97%']}"
            mcse = f"{summary_row['mcse_mean']} ({summary_row['mcse_sd']})"
            rows.append([name, mean_sd, hdi, mcse] + [str(metric) for metric in summary_row[6:]])
        self.add_element_to_doc("Calibration", TableElement("p{1.3cm} " * 7, headers, rows))
        return calib_summary
            
    def add_param_table_to_doc(self):
        """
        Describe all the parameters used in the model, regardless of whether 
        """
        text = "Parameter interpretation, with value (for parameters not included in calibration algorithm) and summary of evidence. \n"
        self.add_element_to_doc("Parameterisation", TextElement(text))

        headers = ["Name", "Value", "Evidence"]
        col_widths = "p{2.7cm} " * 2 + "p{5.8cm}"
        rows = []
        for param in self.model.get_input_parameters():
            param_value_text = get_fixed_param_value_text(param, self.params, self.units, self.prior_names)
            rows.append([self.descriptions[param], param_value_text, NoEscape(self.evidence[param])])
        self.add_element_to_doc("Calibration", TableElement(col_widths, headers, rows))
