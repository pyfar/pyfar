import matplotlib.pyplot as plt
from matplotlib.tight_layout import get_subplotspec_list


def _tight_layout(fig=plt.gcf()):
    """
    Apply Matplotlibs tight_layout only when it is likely to work.

    Tight layout messes up the Figure for irregular subplot layouts. The
    if-case to check if tight layout is applied was taken directly from
    Matplotlib. However, Matplotlib only raises a warning but still applies
    the tight layout.

    Parameters
    ----------
    fig : Matplotlib Figure
    """
    subplotspec_list = get_subplotspec_list(fig.get_axes())
    if None not in subplotspec_list:
        plt.tight_layout()
