from datetime import datetime
import numpy as np
import pandas as pd
from jax import numpy as jnp

import estival.priors as esp
import estival.targets as est

from emutools.tex import StandardTexDoc
from inputs.constants import TARGETS_START_DATE, TARGETS_AVERAGE_WINDOW
from aust_covid.inputs import load_calibration_targets, load_who_data, load_serosurvey_data


def get_priors(vacc_sens: bool) -> list:
    """
    Get the standard priors used for the analysis.

    Args:
        Whether to apply vaccination structure to the model
    
    Returns:
        Final priors
    """
    base_priors = [
        esp.UniformPrior('contact_rate', (0.02, 0.15)),
        esp.GammaPrior.from_mode('latent_period', 2.5, 5.0),
        esp.GammaPrior.from_mode('infectious_period', 3.5, 6.0),
        esp.GammaPrior.from_mode('natural_immunity_period', 180.0, 1000.0),
        esp.UniformPrior('start_cdr', (0.1, 0.6)),
        esp.UniformPrior('imm_infect_protect', (0.0, 1.0)),
        esp.TruncNormalPrior('ifr_adjuster', 1.0, 2.0, (0.2, np.inf)),
        esp.UniformPrior('ba1_seed_time', (580.0, 625.0)), 
        esp.UniformPrior('ba2_seed_time', (625.0, 660.0)),
        esp.UniformPrior('ba5_seed_time', (660.0, 740.0)),
        esp.BetaPrior.from_mean_and_ci('ba2_escape', 0.4, (0.2, 0.6)),
        esp.BetaPrior.from_mean_and_ci('ba5_escape', 0.4, (0.2, 0.6)),
        esp.TruncNormalPrior('ba2_rel_ifr', 0.7, 0.15, (0.2, np.inf)),
        esp.UniformPrior('wa_reopen_period', (30.0, 75.0)),
        esp.GammaPrior.from_mean('notifs_mean', 4.17, 7.0),
        esp.GammaPrior.from_mean('deaths_mean', 15.93, 18.79),
    ]

    if vacc_sens:
        specific_prior = esp.GammaPrior.from_mode('vacc_immune_period', 30.0, 180.0)
    else:
        specific_prior = esp.UniformPrior('imm_prop', (0.0, 1.0))

    return base_priors + [specific_prior]


def truncation_ceiling(modelled, obs, parameters, time_weights):
    """
    Very large negative number to add to likelihood if modelled values 
    above a threshold considered implausible.
    """
    return jnp.where(modelled > obs, -1e11, 0.0)


def get_targets(tex_doc: StandardTexDoc) -> list:
    """
    Get the standard targets used for the analysis.

    Args:
        tex_doc: Supplement documentation object

    Returns:
        Final targets
    """
    description = f'The composite daily case data were then smoothed using a {TARGETS_AVERAGE_WINDOW}-day moving average. '
    tex_doc.add_line(description, 'Targets', 'Notifications')

    case_targets = load_calibration_targets(tex_doc).rolling(window=TARGETS_AVERAGE_WINDOW).mean().dropna()
    death_targets = load_who_data(tex_doc)[TARGETS_START_DATE:].rolling(window=TARGETS_AVERAGE_WINDOW).mean().dropna()
    serosurvey_targets = load_serosurvey_data(tex_doc)
    targets = [
        est.NegativeBinomialTarget('notifications_ma', case_targets, dispersion_param=esp.UniformPrior('notifications_ma_dispersion', (10.0, 140.0))),
        est.NegativeBinomialTarget('deaths_ma', death_targets, dispersion_param=esp.UniformPrior('deaths_ma_dispersion', (60.0, 200.0))),
        est.BinomialTarget('adult_seropos_prop', serosurvey_targets, pd.Series([20] * 4, index=serosurvey_targets.index)),
    ]
    targets.append(est.CustomTarget('seropos_ceiling', pd.Series([0.04], index=[datetime(2021, 12, 1)]), truncation_ceiling, model_key='adult_seropos_prop'))
    return targets
