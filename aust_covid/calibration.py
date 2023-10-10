from datetime import datetime
import numpy as np
import pandas as pd
from jax import numpy as jnp

import estival.priors as esp
import estival.targets as est

from emutools.tex import TexDoc, get_tex_formatted_date
from inputs.constants import TARGETS_START_DATE, TARGETS_AVERAGE_WINDOW
from aust_covid.inputs import load_calibration_targets, load_who_data, load_serosurvey_data


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
    return next((t.data for t in targets if t.name == name))


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
    """See description in get_targets below, standard arguments required.
    """
    return jnp.where(modelled > obs, -1e11, 0.0)


def get_targets(tex_doc: TexDoc) -> list:
    """
    Get the standard targets used for the analysis.

    Args:
        tex_doc: Supplement documentation object

    Returns:
        Final targets
    """
    description = 'Calibration targets were constructed as described throughout the following subsections, ' \
        'and summarised in Figure \\ref{targets}. '
    tex_doc.add_line(description, 'Targets')

    case_targets = load_calibration_targets(tex_doc).rolling(window=TARGETS_AVERAGE_WINDOW).mean().dropna()
    death_targets = load_who_data(tex_doc)[TARGETS_START_DATE:].rolling(window=TARGETS_AVERAGE_WINDOW).mean().dropna()
    serosurvey_targets = load_serosurvey_data(tex_doc)

    description = f'The composite daily case data were then smoothed using a {TARGETS_AVERAGE_WINDOW}-day moving average. ' \
        'The notifications value for each date of the analysis were compared against the modelled estimate ' \
        'from a given parameter set using a negative binomial distribution. The dispersion parameter ' \
        'of this negative binomial distribution was calibrated from an uninformative prior distribution ' \
        'along with the epidemiological parameters through the calibration algorithm. '
    tex_doc.add_line(description, 'Targets', 'Notifications')
    description = f'The WHO data were also smoothed using a {TARGETS_AVERAGE_WINDOW}-day moving average. ' \
        'As for case notifications, the comparison distribution used to obtain the likelihood of a given parameter set ' \
        'was negative binomial with calibrated dispersion parameter. '
    tex_doc.add_line(description, 'Targets', 'Deaths')
    seropos_ceiling = 0.04
    ceiling_date = datetime(2021, 12, 1)
    description = 'The proportion of the population seropositive was compared against ' \
        'the modelled proportion of the population ever infected using a binomial distribution. \n\n' \
        'We added a further recovered proportion target to avoid accepting runs with higher likelihood values ' \
        'in which the acceptable fit to data was a result of an implausibly high initial epidemic wave ' \
        'that occurred prior to the availability of target data (i.e. in late 2021 during the model run-in period). ' \
        'This was achieved by adding a large negative number to the likelihood estimate for any runs with a ' \
        f'proportion ever infected greater than {int(seropos_ceiling * 100)}\% on {get_tex_formatted_date(ceiling_date)}. '
    tex_doc.add_line(description, 'Targets', 'Seroprevalence')

    targets = [
        est.NegativeBinomialTarget('notifications_ma', case_targets, dispersion_param=esp.UniformPrior('notifications_ma_dispersion', (10.0, 140.0))),
        est.NegativeBinomialTarget('deaths_ma', death_targets, dispersion_param=esp.UniformPrior('deaths_ma_dispersion', (60.0, 200.0))),
        est.BinomialTarget('adult_seropos_prop', serosurvey_targets, pd.Series([20] * 4, index=serosurvey_targets.index)),
    ]
    targets.append(est.CustomTarget('seropos_ceiling', pd.Series([seropos_ceiling], index=[ceiling_date]), truncation_ceiling, model_key='adult_seropos_prop'))
    return targets
