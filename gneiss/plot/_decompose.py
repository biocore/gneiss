# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt


def balance_boxplot(balance_name, data, num_color='#FFFFFF',
                    denom_color='#FFFFFF',
                    xlabel="", ylabel="", linewidth=1,
                    ax=None, **kwargs):
    """ Plots a boxplot for a given balance and the associated metadata.

    Parameters
    ----------
    balance_name : str
        Name of balance to plot.
    data : pd.DataFrame
        Merged dataframe of balances and metadata.
    num_color : str
        Hex for background colors of values above zero.
    denom_color : str
        Hex for background colors of values below zero.
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label.
    linewidth : str
        Width of the grid lines.
    ax : matplotlib axes object
        Axes object to render boxplots in.
    **kwargs : dict
        Values to pass in to customize seaborn boxplot.

    Returns
    -------
    a : matplotlib axes object
        Matplotlit axes object with rendered boxplots.

    See Also
    --------
    seaborn.boxplot
    """
    import seaborn as sns
    if ax is None:
        f, ax = plt.subplots()

    # the number 20 is pretty arbitrary - we are just
    # resizing to make sure that there is separation between the
    # edges of the plot, and the boxplot
    pad = (data[balance_name].max() - data[balance_name].min()) / 20
    ax.axvspan(data[balance_name].min()-pad, 0,
               facecolor=num_color, zorder=0)
    ax.axvspan(0, data[balance_name].max()+pad,
               facecolor=denom_color, zorder=0)
    if 'hue' in kwargs.keys():
        num_groups = len(data[hue].value_counts())
    else:
        num_groups = 1
    a = sns.boxplot(ax=ax, y=balance_name, data=data, **kwargs)
    a.minorticks_on()
    minorLocator = matplotlib.ticker.AutoMinorLocator(num_groups)
    a.get_yaxis().set_minor_locator(minorLocator)
    a.grid(axis='y', which='minor', color='k', linestyle=':', linewidth=1)
    a.set_xlim([data[balance_name].min() - pad,
                data[balance_name].max() + pad])
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    return a


def balance_barplots(tree, balance_name, header, feature_metadata,
                     ndim=5, num_color="#0000FF", denom_color="#0000FF",
                     xlabel="", ylabel="",
                     axes=(None, None)):
    """
    Parameters
    ----------
    tree : skbio.TreeNode
        Reference tree for balances.
    balance_name : str
        Name of balance to plot.
    header : str
        Header name for the feature metadata column to summarize
    feature_metadata : pd.DataFrame
        Contains information about the features.
    ndim : int
        Number of bars to display at a given time (default=5)
    num_color : str
        Hex for background colors of values above zero.
    denom_color : str
        Hex for background colors of values below zero.
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label.
    axes : tuple of matplotlib axes objects
        Specifies where the barplots should be rendered.

    Returns
    -------
    ax_num : matplotlib axes object
        Barplot of the features in the numerator of the balance.
    ax_denom : matplotlib axes object
        Barplot of the features in the denominator of the balance.
    """
    import seaborn as sns
    if axes[0] is None or axes[2] is None:
        f, (ax_num, ax_denom) = plt.subplots(2)
    else:
        ax_num, ax_denom = axes[0], axes[1]
    st = tree.find(balance_name)
    num = feature_metadata.loc[list(st.children[0].subset())]
    denom = feature_metadata.loc[list(st.children[1].subset())]

    num_ = num[header].value_counts().head(ndim).reset_index()
    denom_ = denom[header].value_counts().head(ndim).reset_index()

    ax_denom = sns.barplot(y='index', x=header, data=denom_, ax=ax_denom,
                           color=denom_color)
    ax_denom.set_ylabel(ylabel)
    ax_denom.set_xlabel(xlabel)

    ax_denom.set_xlim([0,  max([num_.max().values[1],
                                denom_.max().values[1]])])

    ax_num = sns.barplot(y='index', x=header, data=num_, ax=ax_num,
                         color=num_color)
    ax_num.set_ylabel(ylabel)
    ax_num.set_xlabel(xlabel)
    ax_num.set_xlim([0,  max([num_.max().values[1],
                              denom_.max().values[1]])])
    return ax_num, ax_denom
