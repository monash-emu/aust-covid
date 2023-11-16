from typing import List
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_row_col_for_subplots(i_panel, n_cols):
    return int(np.floor(i_panel / n_cols)) + 1, i_panel % n_cols + 1


def get_standard_subplot_fig(
    n_rows: int, 
    n_cols: int, 
    titles: List[str],
) -> go.Figure:
    """Start a plotly figure with subplots off from standard formatting.

    Args:
        n_rows: Argument to pass through to make_subplots
        n_cols: Pass through
        titles: Pass through

    Returns:
        Figure with nothing plotted
    """
    heights = [320, 600, 900, 900]
    fig = make_subplots(n_rows, n_cols, subplot_titles=titles, vertical_spacing=0.08, horizontal_spacing=0.05)
    return fig.update_layout(margin={i: 25 for i in ['t', 'b', 'l', 'r']}, height=heights[n_rows - 1])
