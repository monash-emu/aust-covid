from typing import Dict, List
import pandas as pd
from datetime import datetime

from inputs.constants import VACC_AVERAGE_WINDOW, VACC_IMMUNE_DURATION


def get_vacc_data_masks(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Get masks for vaccination dataframe to identify the various programs.

    Args:
        df: The vaccination dataframe returned by get_base_vacc_data

    Returns:
        Dictionary containing lists of the names of columns relevant to each program
    """
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
) -> pd.DataFrame:
    """
    Add the additional columns needed in the vaccination dataframe that are not included as raw data.

    Args:
        df: The vaccination dataframe returned by get_base_vacc_data
        rolling_window: Period over which to calculate the rolling average
        immune_duration: Duration of immunity over which to look back for vaccination

    Returns:
        Augmented vaccination dataframe with additional columns
    """
    masks = get_vacc_data_masks(df)
    df['adult booster'] = df.loc[:, masks['age 16+, 3+ doses'] + masks['age 16+, 4+ doses']].sum(axis=1)
    df = df.drop(datetime(2022, 7, 8))
    df['adult booster smooth'] = df.loc[:, 'adult booster'].rolling(VACC_AVERAGE_WINDOW).mean()
    df['incremental adult booster'] = df['adult booster smooth'].diff()
    df['boosted in preceding'] = df['incremental adult booster'].rolling(VACC_IMMUNE_DURATION).sum()
    df['prop boosted in preceding'] = df['boosted in preceding'] / df['National - Population 16 and over']
    return df
