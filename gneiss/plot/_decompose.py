# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from gneiss.util import NUMERATOR, DENOMINATOR


def balance_boxplot(balance_name, data, num_color='#FFFFFF',
                    denom_color='#FFFFFF',
                    xlabel="", ylabel="", linewidth=1,
                    ax=None, **kwargs):
    """ Plots a boxplot for a given balance and the associated metadata.

    Parameters
    ----------
    x, y, hue: str
        Variable names to be passed into the seaborn plots for plotting.
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
        hue = kwargs['hue']
        num_groups = len(data[hue].value_counts())
    else:
        num_groups = 1
    a = sns.boxplot(ax=ax, x=balance_name, data=data, **kwargs)
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
    """ Plots barplots of counts of features found in the balance.

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
    if axes[0] is None or axes[1] is None:
        f, (ax_num, ax_denom) = plt.subplots(2)
    else:
        ax_num, ax_denom = axes[0], axes[1]
    st = tree.find(balance_name)
    num_clade = st.children[NUMERATOR]
    denom_clade = st.children[DENOMINATOR]
    if num_clade.is_tip():
        num_ = pd.DataFrame(
            [[feature_metadata.loc[num_clade.name, header], 1]],
            columns=['index', header],
            index=[header])
    else:
        num = feature_metadata.loc[list(num_clade.subset())]
        num_ = num[header].value_counts().head(ndim).reset_index()

    if denom_clade.is_tip():
        denom_ = pd.DataFrame(
            [[feature_metadata.loc[denom_clade.name, header], 1]],
            columns=['index', header],
            index=[header])
    else:
        denom = feature_metadata.loc[list(denom_clade.subset())]
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


def proportion_plot(table, metadata, num_features, denom_features,
                    feature_metadata, category, left_group, right_group,
                    taxa_level='species',
                    num_color='#105d33', denom_color='#b0d78e',
                    axes=(None, None)):
    """ Plot the mean proportions of features within a balance.

    This plots the numerator and denominator components within a balance.

    Parameters
    ----------
    table : pd.DataFrame
       Table of relative abundances.
    metadata : pd.DataFrame
       Samples metadata
    spectrum : pd.Series
       The component from partial least squares.
    feature_metadata : pd.DataFrame
       The metadata associated to the features.
    category : str
       Name of sample metadata category.
    left_group : str
       Name of group within sample metadata category to plot
       on the left of the plot.
    right_group : str
       Name of group within sample metadata category to plot
       on the right of the plot.
    taxa_level : str
       Taxonomic level to summarize.
    num_color : str
       Color to plot the numerator.
    denom_color : str
       Color to plot the denominator.
    """
    import seaborn as sns
    if axes[0] is None or axes[1] is None:
        f, (ax_num, ax_denom) = plt.subplots(1, 2)
    else:
        ax_num, ax_denom = axes[0][0], axes[0][1]

    level = 'feature'
    ptable = table.apply(lambda x: (x+1) / (x+1).sum(), axis=1)
    num_collapsed = ptable[num_features]

    denom_collapsed = ptable[denom_features]

    # merge together metadata and sequences
    num_data_ = pd.merge(metadata, num_collapsed,
                         left_index=True, right_index=True)
    denom_data_ = pd.merge(metadata, denom_collapsed,
                           left_index=True, right_index=True)

    # merge the data frame, so that all of the proportions
    # are in their own separate column
    num_data = pd.melt(num_data_, id_vars=[category],
                       value_vars=list(num_collapsed.columns),
                       value_name='proportion', var_name=level)
    num_data['part'] = 'numerator'
    denom_data = pd.melt(denom_data_, id_vars=[category],
                         value_vars=list(denom_collapsed.columns),
                         value_name='proportion', var_name=level)
    denom_data['part'] = 'denominator'
    data = pd.concat((num_data, denom_data))

    num_feature_metadata = feature_metadata.loc[num_collapsed.columns,
                                                taxa_level]
    denom_feature_metadata = feature_metadata.loc[denom_collapsed.columns,
                                                  taxa_level]

    less_df = data.loc[data[category] == left_group].dropna()
    # order of the ids to plot
    order = (list(num_feature_metadata.index) +
             list(denom_feature_metadata.index))
    sns.barplot(x='proportion',
                y=level,
                data=less_df,
                color=denom_color,
                order=order,
                ax=ax_denom)
    more_df = data.loc[data[category] == right_group].dropna()

    sns.barplot(x='proportion',
                y=level,
                data=more_df,
                color=num_color,
                order=order,
                ax=ax_num)

    ax_denom.set(yticklabels=(list(num_feature_metadata.values) +
                              list(denom_feature_metadata.values)),
                 title=left_group)
    ax_num.set(yticklabels=[], ylabel='', yticks=[], title=right_group)

    max_xlim = max(ax_denom.get_xlim()[1], ax_num.get_xlim()[1])
    min_xlim = max(ax_denom.get_xlim()[0], ax_num.get_xlim()[0])

    max_ylim, min_ylim = ax_denom.get_ylim()

    xlim = ([min_xlim, max_xlim])
    ax_denom.set_xlim(max_xlim, min_xlim)
    ax_num.set_xlim(min_xlim, max_xlim)
    ax_denom.set_position([0.2, 0.125, 0.3, 0.75])
    ax_num.set_position([0.5, 0.125, 0.3, 0.75])

    num_h = num_collapsed.shape[1]
    denom_h = denom_collapsed.shape[1]

    space = (max_ylim - min_ylim) / (num_h + denom_h)
    ymid = (max_ylim - min_ylim) * num_h / (num_h + denom_h) - 0.5 * space

    ax_denom.axhspan(min_ylim, ymid, facecolor=num_color,
                     zorder=0, alpha=0.25)
    ax_denom.axhspan(ymid, max_ylim, facecolor=denom_color,
                     zorder=0, alpha=0.25)

    ax_num.axhspan(min_ylim, ymid, facecolor=num_color, zorder=0, alpha=0.25)
    ax_num.axhspan(ymid, max_ylim, facecolor=denom_color, zorder=0, alpha=0.25)
    return (ax_num, ax_denom)
