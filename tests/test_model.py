from pathlib import Path
from datetime import datetime
from aust_covid import model
import pandas as pd
pd.options.plotting.backend = 'plotly'
from emutools.tex import StandardTexDoc
from aust_covid.inputs import get_ifrs

PROJECT_PATH = Path().resolve()


def test_smoke_model():
    ref_date = datetime(2019, 12, 31)
    start_date = datetime(2021, 9, 1)
    end_date = datetime(2022, 10, 1)
    parameters = {
        'contact_rate': 0.065,
        'latent_period': 1.8,
        'infectious_period': 2.5,
        'natural_immunity_period': 60.0,
        'start_cdr': 0.3,
        'imm_prop': 0.4,
        'imm_infect_protect': 0.4,
        'ifr_adjuster': 3.0,
        'ba1_seed_time': 619.0,
        'ba2_seed_time': 660.0,
        'ba5_seed_time': 715.0,
        'ba2_escape': 0.4,
        'ba5_escape': 0.54,
        'ba2_rel_ifr': 0.5,
        'seed_duration': 10.0,
        'seed_rate': 1.0,
        'wa_reopen_period': 30.0,
        'notifs_mean': 4.0,
        'notifs_shape': 2.0,
        'deaths_mean': 20.0,
        'deaths_shape': 2.0,
    }
    app_doc = StandardTexDoc(PROJECT_PATH / 'supplement', 'supplement', "Australia's 2023 Omicron Waves Supplement", 'austcovid')
    ifrs = get_ifrs(app_doc)
    parameters.update(ifrs) 
    aust_model = model.build_model(ref_date, start_date, end_date, app_doc, 7)
    aust_model.run(parameters=parameters)
