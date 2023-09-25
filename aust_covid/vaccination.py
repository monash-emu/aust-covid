from typing import Dict, List
import pandas as pd
from datetime import datetime


def get_vacc_data_masks(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    masks = {
        'age 16+, 2+ doses': [
            c for c in df.columns if 'Number of people fully vaccinated' in c and 
            not any(s in c for s in [' - M - ', ' - F - '])
        ],
        'age 16+, 3+ doses': [
            col for col in df.columns if 
            'National' in col and 
            any([s in col for s in ['who have received 3 doses', 'with 3 or more doses', 'with more than two doses', 'with more than 2 doses']]) and 
            any([s in col for s in ['16', '18']]) and
            not any([s in col for s in ['increase', 'Percentage', 'Indigenous']])
        ],
        'age 16+, 4+ doses': [
            col for col in df.columns if
            'National' in col and
            any([s in col for s in ['Winter Boosters number', 'who have received 4 doses', 'Fourth dose number']]) and
            not any([s in col for s in ['30', '65', 'increase']])
        ],
        'age 12-15, 2+ doses': [
            col for col in df.columns if
            '12-15' in col and
            any([s in col for s in ['National', 'Age group']]) and
            any([s in col for s in ['2 doses', 'fully vaccinated']]) and
            not any([s in col for s in ['Indigenous', 'Population', '- F -', '- M -']])
        ],
        'age 5-11, 2+ doses': [
            col for col in df.columns if
            'National' in col and
            '5-11' in col and
            any([s in col for s in ['2 doses', 'fully vaccinated']])
        ],
    }
    masks['age 16+, 2+ doses'].sort()
    return masks


def add_booster_data_to_vacc(
    df: pd.DataFrame, 
    rolling_window: int, 
    immune_duration: int,
) -> pd.DataFrame:
    masks = get_vacc_data_masks(df)
    df['adult booster'] = df.loc[:, masks['age 16+, 3+ doses'] + masks['age 16+, 4+ doses']].sum(axis=1)
    df = df.drop(datetime(2022, 7, 8))
    df['adult booster smooth'] = df.loc[:, 'adult booster'].rolling(rolling_window).mean()
    df['incremental adult booster'] = df['adult booster smooth'].diff()
    df['boosted in preceding'] = df['incremental adult booster'].rolling(immune_duration).sum()
    df['prop boosted in preceding'] = df['boosted in preceding'] / df['National - Population 16 and over']
    return df
