from aust_covid.model import build_aust_model
from datetime import datetime
import pylatex as pl


def test_smoke_model():
    supplement = pl.Document()
    start_date = datetime(2021, 8, 22)
    end_date = datetime(2022, 6, 10)
    parameters = {
        "contact_rate": 0.048,
        "infectious_period": 5.0,
        "latent_period": 2.0,
        "cdr": 0.1,
        "seed_rate": 1.0,
        "seed_duration": 1.0,
        "ba1_seed_time": 660.0,
        "ba2_seed_time": 688.0,
        "ba5_seed_time": 720.0,
        "ba2_escape": 0.45,
        "ba5_escape": 0.38,
        "notifs_shape": 2.0,
        "notifs_mean": 4.0,
        "deaths_shape": 2.0,
        "deaths_mean": 20.0,
        "natural_immunity_period": 50.0,
        "ifr_0": 0.0,
        "ifr_5": 0.0,
        "ifr_10": 0.0,
        "ifr_15": 0.0,
        "ifr_20": 0.0,
        "ifr_25": 0.0,
        "ifr_30": 0.0,
        "ifr_35": 0.0,
        "ifr_40": 0.0,
        "ifr_45": 0.0,
        "ifr_50": 0.0,
        "ifr_55": 0.0,
        "ifr_60": 0.0,
        "ifr_65": 0.0,
        "ifr_70": 0.01,
    }
    aust_model = build_aust_model(start_date, end_date, supplement, add_documentation=False)
    aust_model.run(parameters=parameters)

