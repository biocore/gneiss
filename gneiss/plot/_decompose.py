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
    """ Plots a boxplot for a given balance on a discrete metadata category.

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
    """ Plots barplots of features found in a given balance.
        These are the most abundant taxa on each side of the balance
        (numerator & denominator).  The x-axis is the number of features
        within the taxonomic classification denoted on the y-axis.

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
        Number of bars (features) to display at a given time (default=5)
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


def proportion_plot(table, metadata, category, left_group, right_group,
                    num_features, denom_features,
                    feature_metadata=None,
                    label_col='species',
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
    label_col : str
       Column in the feature metadata table to summarize by.
    num_color : str
       Color to plot the numerator.
    denom_color : str
       Color to plot the denominator.

    Returns
    -------
    ax_num : matplotlib.pyplot.Axes
       Matplotlib axes for the numerator bars
    ax_denom : matplotlib.pyplot.Axes
       Matplotlib axes for the denominator bars

    Examples
    --------
    First we'll want to set up the main objects to pass into the plot.
    For starters, we'll pass in the feature table, metadata and
    feature_metadata.

    >>> table = pd.DataFrame({
    ... 'A': [1, 1.2, 1.1, 2.1, 2.2, 2],
    ... 'B': [9.9, 10, 10.1, 2, 2.4, 2.1],
    ... 'C': [5, 3, 1, 2, 2, 3],
    ... 'D': [5, 5, 5, 5, 5, 5],
    ... }, index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

    >>> feature_metadata = pd.DataFrame({
    ...     'A': ['k__foo', 'p__bar', 'c__', 'o__', 'f__', 'g__', 's__'],
    ...     'B': ['k__foo', 'p__bar', 'c__', 'o__', 'f__', 'g__', 's__'],
    ...     'C': ['k__poo', 'p__tar', 'c__', 'o__', 'f__', 'g__', 's__'],
    ...     'D': ['k__poo', 'p__far', 'c__', 'o__', 'f__', 'g__', 's__']
    ... }, index=['kingdom', 'phylum', 'class', 'order', 'family',
    ...           'genus', 'species']).T

    >>> metadata = pd.DataFrame({
    ...     'groups': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
    ...     'dry': [1, 2, 3, 4, 5, 6]},
    ...     index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

    Then we can specify which specific features to visualize and plot.
    >>> num_features = ['A', 'B']
    >>> denom_features = ['C', 'D']
    >>> ax1, ax2 = proportion_plot(table, metadata, 'groups', 'X', 'Y',
    ...                            num_features, denom_features,
    ...                            feature_metadata, label_col='phylum')

    Since this method will return the raw matplotlib object, labels, titles,
    ticks, etc can directly modified using this object.
    """
    import seaborn as sns
    if axes[0] is None or axes[1] is None:
        f, (ax_num, ax_denom) = plt.subplots(1, 2)
    else:
        ax_num, ax_denom = axes[0], axes[1]

    fname = 'feature'
    ptable = table.apply(lambda x: x / x.sum(), axis=1)
    num_df = ptable[num_features]

    denom_df = ptable[denom_features]

    # merge together metadata and sequences
    num_data_ = pd.merge(metadata, num_df,
                         left_index=True, right_index=True)
    denom_data_ = pd.merge(metadata, denom_df,
                           left_index=True, right_index=True)

    # merge the data frame, so that all of the proportions
    # are in their own separate column
    num_data = pd.melt(num_data_, id_vars=[category],
                       value_vars=list(num_df.columns),
                       value_name='proportion', var_name=fname)
    num_data['part'] = 'numerator'
    denom_data = pd.melt(denom_data_, id_vars=[category],
                         value_vars=list(denom_df.columns),
                         value_name='proportion', var_name=fname)
    denom_data['part'] = 'denominator'
    data = pd.concat((num_data, denom_data))
    if feature_metadata is not None:
        num_feature_metadata = feature_metadata.loc[num_df.columns,
                                                    label_col]
        denom_feature_metadata = feature_metadata.loc[denom_df.columns,
                                                      label_col]
        # order of the ids to plot
        order = (list(num_feature_metadata.index) +
                 list(denom_feature_metadata.index))
    else:
        order = (list(num_df.columns) +
                 list(denom_df.columns))

    less_df = data.loc[data[category] == left_group].dropna()

    sns.barplot(x='proportion',
                y=fname,
                data=less_df,
                color=denom_color,
                order=order,
                ax=ax_denom)
    more_df = data.loc[data[category] == right_group].dropna()

    sns.barplot(x='proportion',
                y=fname,
                data=more_df,
                color=num_color,
                order=order,
                ax=ax_num)
    if feature_metadata is not None:
        ax_denom.set(yticklabels=(list(num_feature_metadata.values) +
                                  list(denom_feature_metadata.values)),
                     title=left_group)
    else:
        ax_denom.set(yticklabels=order, title=left_group)

    ax_num.set(yticklabels=[], ylabel='', yticks=[], title=right_group)

    max_xlim = max(ax_denom.get_xlim()[1], ax_num.get_xlim()[1])
    min_xlim = max(ax_denom.get_xlim()[0], ax_num.get_xlim()[0])

    max_ylim, min_ylim = ax_denom.get_ylim()

    ax_denom.set_xlim(max_xlim, min_xlim)
    ax_num.set_xlim(min_xlim, max_xlim)
    ax_denom.set_position([0.2, 0.125, 0.3, 0.75])
    ax_num.set_position([0.5, 0.125, 0.3, 0.75])

    num_h = num_df.shape[1]
    denom_h = denom_df.shape[1]

    space = (max_ylim - min_ylim) / (num_h + denom_h)
    ymid = (max_ylim - min_ylim) * num_h / (num_h + denom_h) - 0.5 * space

    ax_denom.axhspan(min_ylim, ymid, facecolor=num_color,
                     zorder=0, alpha=0.25)
    ax_denom.axhspan(ymid, max_ylim, facecolor=denom_color,
                     zorder=0, alpha=0.25)

    ax_num.axhspan(min_ylim, ymid, facecolor=num_color, zorder=0, alpha=0.25)
    ax_num.axhspan(ymid, max_ylim, facecolor=denom_color, zorder=0, alpha=0.25)
    return (ax_num, ax_denom)
