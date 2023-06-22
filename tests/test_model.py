from pathlib import Path
from datetime import datetime
from general_utils.parameter_utils import load_param_info
from aust_covid import model
import pandas as pd
pd.options.plotting.backend = "plotly"

PROJECT_PATH = Path().resolve()


def test_smoke_model():
    start_date = datetime(2021, 9, 1)
    end_date = datetime(2022, 10, 1)
    plot_start_date = datetime(2021, 12, 1)  # Left end for plots
    ref_date = datetime(2019, 12, 31)
    parameters = {
        "contact_rate": 0.0458, 
        "infectious_period": 6.898, 
        "start_cdr": 0.0997, 
        "ba1_seed_time": 657.4, 
        "natural_immunity_period": 23.5,
        "ba2_seed_time": 700.0,
        "ba2_escape": 0.8,
        "ba5_seed_time": 765., 
        "ba5_escape": 1.0,
        "latent_period": 2.0,
        "seed_rate": 1.0,
        "seed_duration": 1.0,
        "notifs_shape": 2.0,
        "notifs_mean": 4.0,
    }
    param_info = load_param_info(PROJECT_PATH / "inputs/parameters.yml", parameters)
    compartments = [
        "susceptible",
        "latent",
        "infectious",
        "recovered",
        "waned",
    ]    
    aust_model, _ = model.build_base_model(ref_date, compartments, start_date, end_date)
    pop_data, _ = model.get_pop_data()
    model.set_starting_conditions(aust_model, pop_data, adjuster=1.0)
    model.add_infection(aust_model)
    model.add_progression(aust_model)
    model.add_recovery(aust_model)
    model.add_waning(aust_model)
    age_strata = list(range(0, 80, 5))
    raw_mob_df, _, _, _ = model.get_raw_mobility(start_date, aust_model)
