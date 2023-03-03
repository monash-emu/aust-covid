from aust_covid.model import build_aust_model
from datetime import datetime
import pylatex as pl


def test_smoke_model():
    supplement = pl.Document()
    start_date = datetime(2021, 8, 22)
    end_date = datetime(2022, 6, 10)
    parameters = {
        "contact_rate": 0.028,
        "infectious_period": 5.0,
        "latent_period": 1.5,
        "full_immune_period": 30.0,
        "cdr": 0.2,
        "ba2_rel_infness": 1.8,
        "seed_rate": 1.0,
        "seed_duration": 1.0,
        "ba1_seed_time": 600.0,
        "ba2_seed_time": 720.0,
        "ba1infection_protect_ba1": 1.0,
        "ba1infection_protect_ba2": 0.0,
        "ba2infection_protect_ba1": 1.0,
        "ba2infection_protect_ba2": 1.0,
    }
    aust_model = build_aust_model(start_date, end_date, supplement, add_documentation=False)
    aust_model.run(parameters=parameters)

