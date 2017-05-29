# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skbio import TreeNode
from skbio.stats.composition import clr, centralize

from gneiss.plugin_setup import plugin
from gneiss.plot._heatmap import heatmap
from gneiss.plot._decompose import balance_barplots, balance_boxplot
from gneiss.util import (match, NUMERATOR, DENOMINATOR)

from q2_types.tree import Phylogeny, Rooted
from q2_composition.plugin_setup import Composition, Balance
from q2_types.feature_table import FeatureTable
from q2_types.feature_data import FeatureData, Taxonomy
from qiime2.plugin import Int, MetadataCategory, Str, Choices

_taxa_headers = ['1', '2', '3', '4', '5', '6', '7'],


def balance_taxonomy(output_dir: str, balances: pd.DataFrame, tree: TreeNode,
                     taxonomy: pd.DataFrame,
                     balance_name: Str,
                     taxa_level: Str = 'phyla',
                     metadata: MetadataCategory = None) -> None:

    # parse out headers for taxonomy
    taxa_data = list(taxonomy['Taxon'].apply(lambda x: x.split(';')).values)
    taxa_df = pd.DataFrame(taxa_data, columns=_taxa_headers,
                           index=taxonomy.index)

    # fill in NAs
    def f(x):
        y = np.array(list(map(lambda k: k is not None, x)))
        i = max(0, np.where(y)[0][-1])
        x[np.logical_not(y)] = [x[i]] * np.sum(np.logical_not(y))
        return x
    taxa_df = taxa_df.apply(f, axis=1)

    num_clade = tree.find(balance_name).children[NUMERATOR]
    denom_clade = tree.find(balance_name).children[DENOMINATOR]

    if num_clade.is_tip():
        num_features = pd.DataFrame(
            {num_clade.name: taxa_df.loc[num_clade.name]}
            ).T
    else:
        num_features = taxa_df.loc[num_clade.subset()]

    if denom_clade.is_tip():
        denom_features = pd.DataFrame(
            {denom_clade.name: taxa_df.loc[denom_clade.name]}
            ).T
    else:
        denom_features = taxa_df.loc[denom_clade.subset()]

    fig, (ax_num, ax_denom) = plt.subplots(2)
    balance_barplots(tree, balance_name, taxa_level, taxa_df,
                     axes=(ax_num, ax_denom))

    ax_num.set_title('Numerator taxa (%d taxa)' % len(num_features))
    ax_denom.set_title('Denominator taxa (%d taxa)' % len(denom_features))
    ax_denom.set_xlabel('Number of taxa')
    fig.savefig(os.path.join(output_dir, 'barplots.svg'))

    if metadata is not None:
        fig2, ax = plt.subplots()
        c = metadata.to_series()
        data, c = match(balances, c)
        data[c.name] = c
        y = data[balance_name]
        # check if continuous
        try:
            c = c.astype(np.float64)
            ax.scatter(c.values, y)
            ax.set_xlabel(c.name)
            ax.set_ylabel(balance_name)
        except:
            balance_boxplot(balance_name, data, y=c.name, ax=ax)
        fig2.savefig(os.path.join(output_dir, 'balance_metadata.svg'))

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        if metadata is not None:
            index_f.write('<h1>Balance vs %s </h1>\n' % c.name)
            index_f.write(('<img src="balance_metadata.svg" '
                           'alt="barplots">\n\n'))

        index_f.write('<h1>Balance Taxonomy</h1>\n')
        index_f.write('<img src="barplots.svg" alt="barplots">\n\n')
        index_f.write(('<h3>Numerator taxa</h3>\n'
                       '<a href="numerator.csv">\n'
                       'Download as CSV</a><br>\n'
                       '<h3>Denominator taxa</h3>\n'
                       '<a href="denominator.csv">\n'
                       'Download as CSV</a><br>\n'))

        num_features.to_csv(os.path.join(output_dir, 'numerator.csv'),
                            header=True, index=True)
        denom_features.to_csv(os.path.join(output_dir, 'denominator.csv'),
                              header=True, index=True)
        index_f.write('</body></html>\n')


plugin.visualizers.register_function(
    function=balance_taxonomy,
    inputs={'balances': FeatureTable[Balance], 'tree': Phylogeny[Rooted],
            'taxonomy': FeatureData[Taxonomy]},
    parameters={'balance_name': Str,
                'taxa_level': Str % Choices(_taxa_headers),
                'metadata': MetadataCategory},
    input_descriptions={
        'balances': 'The table of balances resulting from the ilr transform.',
        'tree': 'The tree used to calculate the balances.',
        'taxonomy': 'Taxonomy information for the OTUs.',
    },
    parameter_descriptions={
        'balance_name': 'Name of the balance to summarize.',
        'taxa_level': 'Level of taxonomy to summarize.',
        'metadata': 'Metadata for plotting the balance (optional).'},
    name='Balance Summary',
    description=("Visualize the distribution of a single balance "
                 "and summarize its numerator and denominator components.")
)


_transform_methods = ['clr', 'log']
_mpl_colormaps = ['viridis', 'inferno', 'plasma', 'magma',
                  'Blues', 'BuGn', 'BuPu',
                  'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                  'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                  'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd',
                  'afmhot', 'autumn', 'bone', 'cool',
                  'copper', 'gist_heat', 'gray', 'hot',
                  'pink', 'spring', 'summer', 'winter',
                  'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                  'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                  'seismic', 'Accent', 'Dark2', 'Paired', 'Pastel1',
                  'Pastel2', 'Set1', 'Set2', 'Set3', 'Vega10',
                  'Vega20', 'Vega20b', 'Vega20c',
                  'gist_earth', 'terrain', 'ocean', 'gist_stern',
                  'brg', 'CMRmap', 'cubehelix',
                  'gnuplot', 'gnuplot2', 'gist_ncar',
                  'nipy_spectral', 'jet', 'rainbow',
                  'gist_rainbow', 'hsv', 'flag', 'prism']


# Heatmap
def dendrogram_heatmap(output_dir: str, table: pd.DataFrame,
                       tree: TreeNode, metadata: MetadataCategory,
                       ndim=10, method='clr', color_map='viridis'):

    nodes = [n.name for n in tree.levelorder() if not n.is_tip()]

    nlen = min(ndim, len(nodes))
    highlights = pd.DataFrame([['#00FF00', '#FF0000']] * nlen,
                              index=nodes[:nlen])
    if method == 'clr':
        mat = pd.DataFrame(clr(centralize(table)),
                           index=table.index,
                           columns=table.columns)
    elif method == 'log':
        mat = pd.DataFrame(np.log(table),
                           index=table.index,
                           columns=table.columns)

    # TODO: There are a few hard-coded constants here
    # will need to have some adaptive defaults set in the future
    fig = heatmap(mat, tree, metadata.to_series(), highlights, cmap=color_map,
                  highlight_width=0.01, figsize=(12, 8))
    fig.savefig(os.path.join(output_dir, 'heatmap.svg'))

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Dendrogram heatmap</h1>\n')
        index_f.write('<img src="heatmap.svg" alt="heatmap">')
        index_f.write('</body></html>\n')


plugin.visualizers.register_function(
    function=dendrogram_heatmap,
    inputs={'table': FeatureTable[Composition],
            'tree': Phylogeny[Rooted]},
    parameters={'metadata': MetadataCategory, 'ndim': Int,
                'method': Str % Choices(_transform_methods),
                'color_map': Str % Choices(_mpl_colormaps)},
    input_descriptions={
        'table': ('The feature table that will be plotted as a heatmap. '
                  'This table is assumed to have strictly positive values.'),
        'tree': ('A hierarchy of feature identifiers where each tip'
                 'corresponds to the feature identifiers in the table. '
                 'This tree can contain tip ids that are not present in '
                 'the table, but all feature ids in the table must be '
                 'present in this tree.')},
    parameter_descriptions={
        'metadata': ('Metadata to group the samples. '),
        'ndim': 'Number of dimensions to highlight.',
        'method': ("Specifies how the data should be normalized for display."
                   "Options include 'log' or 'clr' (default='clr')."),
        'color_map': ("Specifies the color map for plotting the heatmap. "
                      "See https://matplotlib.org/examples/color/"
                      "colormaps_reference.html for more details.")
    },
    name='Dendrogram heatmap.',
    description=("Visualize the feature tables as a heatmap. "
                 "with samples sorted along a specified metadata category "
                 "and features clustered together specified by the tree.")
)
