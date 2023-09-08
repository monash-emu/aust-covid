from datetime import datetime
import numpy as np
import pandas as pd
from jax import numpy as jnp

import estival.priors as esp
import estival.targets as est

from aust_covid.inputs import load_calibration_targets, load_who_data, load_serosurvey_data


def get_priors():
    return [
        esp.UniformPrior('contact_rate', (0.02, 0.15)),
        esp.GammaPrior.from_mode('latent_period', 2.5, 5.0),
        esp.GammaPrior.from_mode('infectious_period', 3.5, 6.0),
        esp.GammaPrior.from_mode('natural_immunity_period', 180.0, 1000.0),
        esp.UniformPrior('start_cdr', (0.1, 0.6)),
        esp.UniformPrior('imm_prop', (0.0, 1.0)),
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


def truncation_ceiling(modelled, obs, parameters, time_weights):
    return jnp.where(modelled > obs, -1e11, 0.0)


def get_targets(app_doc):
    case_targets = load_calibration_targets(app_doc)
    death_targets = load_who_data(app_doc)
    serosurvey_targets = load_serosurvey_data(app_doc)
    targets = [
        est.NegativeBinomialTarget('notifications_ma', case_targets, dispersion_param=esp.UniformPrior('notifications_ma_dispersion', (10.0, 140.0))),
        est.NegativeBinomialTarget('deaths_ma', death_targets, dispersion_param=esp.UniformPrior('deaths_ma_dispersion', (60.0, 200.0))),
        est.BinomialTarget('adult_seropos_prop', serosurvey_targets, pd.Series([20] * 4, index=serosurvey_targets.index)),
    ]
    targets.append(est.CustomTarget('seropos_ceiling', pd.Series([0.04], index=[datetime(2021, 12, 1)]), truncation_ceiling, model_key='adult_seropos_prop'))
    return targets
