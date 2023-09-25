from typing import Dict, List
import pandas as pd
from datetime import datetime


def get_vacc_data_masks(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    masks = {
        'age_16_2dose': [
            c for c in df.columns if 'Number of people fully vaccinated' in c and 
            not any(s in c for s in [' - M - ', ' - F - '])
        ],
        'age_16_3dose': [
            col for col in df.columns if 
            'National' in col and 
            any([s in col for s in ['who have received 3 doses', 'with 3 or more doses', 'with more than two doses', 'with more than 2 doses']]) and 
            any([s in col for s in ['16', '18']]) and
            not any([s in col for s in ['increase', 'Percentage', 'Indigenous']])
        ],
        'age_16_4dose': [
            col for col in df.columns if
            'National' in col and
            any([s in col for s in ['Winter Boosters number', 'who have received 4 doses', 'Fourth dose number']]) and
            not any([s in col for s in ['30', '65', 'increase']])
        ],
        'age_12_15_2dose': [
            col for col in df.columns if
            '12-15' in col and
            any([s in col for s in ['National', 'Age group']]) and
            any([s in col for s in ['2 doses', 'fully vaccinated']]) and
            not any([s in col for s in ['Indigenous', 'Population', '- F -', '- M -']])
        ],
        'age_5_11_2dose': [
            col for col in df.columns if
            'National' in col and
            '5-11' in col and
            any([s in col for s in ['2 doses', 'fully vaccinated']])
        ],
    }
    masks['age_16_2dose'].sort()
    return masks


def add_booster_data_to_vacc(
    df: pd.DataFrame, 
    rolling_window: int, 
    immune_duration: int,
) -> pd.DataFrame:
    masks = get_vacc_data_masks(df)
    df['adult booster'] = df.loc[:, masks['age_16_3dose'] + masks['age_16_4dose']].sum(axis=1)
    df = df.drop(datetime(2022, 7, 8))
    df['adult booster smooth'] = df.loc[:, 'adult booster'].rolling(rolling_window).mean()
    df['incremental adult booster'] = df['adult booster smooth'].diff()
    df['boosted in preceding'] = df['incremental adult booster'].rolling(immune_duration).sum()
    df['prop boosted in preceding'] = df['boosted in preceding'] / df['National - Population 16 and over']
    return df
