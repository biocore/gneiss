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
from skbio import TreeNode

from gneiss.plot._radial import radialplot
from gneiss.regression._ols import OLSModel
from gneiss.regression._mixedlm import LMEModel

from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row, column
from bokeh.models import (HoverTool, BoxZoomTool, WheelZoomTool,
                          ResetTool, SaveTool, PanTool,
                          FuncTickFormatter, FixedTicker)
from bokeh.palettes import RdYlBu11 as palette
from statsmodels.sandbox.stats.multicomp import multipletests


def _projected_prediction(model, plot_width=400, plot_height=400):
    """ Create projected prediction plot

    Parameters
    ----------
    model : RegressionModel
        Input regression model to plot prediction.
    plot_width : int, optional
        Width of plot.
    plot_height : int, optional
        Height of plot.

    Returns
    -------
    bokeh plot
    """
    hover = HoverTool(
            tooltips=[
                ("#SampleID", "@index"),
            ]
        )

    pred = model.predict()
    pcvar = model.percent_explained()
    pred['color'] = 'predicted'  # make predictions red
    raw = model.response_matrix
    raw['color'] = 'raw'   # make raw values blue
    p = figure(plot_width=plot_width, plot_height=plot_height,
               tools=[hover, BoxZoomTool(), ResetTool(),
                      WheelZoomTool(), SaveTool(), PanTool()])
    raw_source = ColumnDataSource(raw)
    pred_source = ColumnDataSource(pred)

    p.circle(raw.columns[0], raw.columns[1], size=7,
             source=raw_source, fill_color='blue', legend='raw')
    p.circle(pred.columns[0], pred.columns[1], size=7,
             source=pred_source, fill_color='red', legend='predicted')

    p.title.text = 'Projected Prediction'
    p.title_location = 'above'

    p.xaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[0], pcvar.iloc[0])
    p.yaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[1], pcvar.iloc[1])
    return p


def _projected_residuals(model, plot_width=400, plot_height=400):
    """ Create projected residual plot

    Parameters
    ----------
    model : RegressionModel
        Input regression model to plot prediction.
    plot_width : int, optional
        Width of plot.
    plot_height : int, optional
        Height of plot.

    Returns
    -------
    bokeh plot
    """
    hover = HoverTool(
            tooltips=[
                ("#SampleID", "@index"),
            ]
        )
    pcvar = model.percent_explained()
    resid = model.residuals()
    p = figure(plot_width=plot_width, plot_height=plot_height,
               tools=[hover, BoxZoomTool(), ResetTool(),
                      WheelZoomTool(), SaveTool(), PanTool()])
    resid_source = ColumnDataSource(resid)

    p.circle(resid.columns[0], resid.columns[1], size=7,
             source=resid_source, fill_color='blue', legend='residuals')

    p.title.text = 'Projected Residuals'
    p.title_location = 'above'
    p.xaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[0], pcvar.iloc[0])
    p.yaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[1], pcvar.iloc[1])
    return p


def _heatmap_summary(pvals, coefs, plot_width=1200, plot_height=400):
    """ Plots heatmap of coefficients colored by pvalues

    Parameters
    ----------
    pvals : pd.DataFrame
        Table of pvalues where rows are balances and columns are
        covariates.
    coefs : pd.DataFrame
        Table of coefficients where rows are balances and columns are
        covariates.
    plot_width : int, optional
        Width of plot.
    plot_height : int, optional
        Height of plot.

    Returns
    -------
    bokeh.charts.Heatmap
        Heatmap summarizing the regression statistics.
    """
    c = coefs.reset_index()
    c = c.rename(columns={'index': 'balance'})

    # fix alpha in fdr to account for the number of covariates
    def fdr(x):
        return multipletests(x, method='fdr_bh',
                             alpha=0.05 / pvals.shape[1])[1]
    cpvals = pvals.apply(fdr, axis=0)

    # log scale for coloring
    log_p = -np.log10(cpvals+1e-200)
    log_p = log_p.reset_index()
    log_p = log_p.rename(columns={'index': 'balance'})
    p = pvals.reset_index()
    p = p.rename(columns={'index': 'balance'})

    cp = cpvals.reset_index()
    cp = cp.rename(columns={'index': 'balance'})

    cm = pd.melt(c, id_vars='balance', var_name='Covariate',
                 value_name='Coefficient')
    pm = pd.melt(p, id_vars='balance', var_name='Covariate',
                 value_name='Pvalue')
    cpm = pd.melt(cp, id_vars='balance', var_name='Covariate',
                  value_name='Corrected_Pvalue')
    logpm = pd.melt(log_p, id_vars='balance', var_name='Covariate',
                    value_name='log_Pvalue')
    m = pd.merge(cm, pm,
                 left_on=['balance', 'Covariate'],
                 right_on=['balance', 'Covariate'])
    m = pd.merge(m, logpm,
                 left_on=['balance', 'Covariate'],
                 right_on=['balance', 'Covariate'])
    m = pd.merge(m, cpm,
                 left_on=['balance', 'Covariate'],
                 right_on=['balance', 'Covariate'])

    hover = HoverTool(
        tooltips=[("Pvalue", "@Pvalue"),
                  ("Corrected Pvalue", "@Corrected_Pvalue"),
                  ("Coefficient", "@Coefficient")]
    )

    N, _min, _max = len(palette), m.log_Pvalue.min(), m.log_Pvalue.max()
    X = pd.Series(np.arange(len(pvals.index)), index=pvals.index)
    Y = pd.Series(np.arange(len(pvals.columns)), index=pvals.columns)
    m['X'] = [X.loc[i] for i in m.balance]
    m['Y'] = [Y.loc[i] for i in m.Covariate]

    # fill in nans with zero.  Sometimes the pvalue calculation fails.
    m = m.fillna(0)
    for i in m.index:
        x = m.loc[i, 'log_Pvalue']
        ind = int(np.floor((x - _min) / (_max - _min) * (N - 1)))
        m.loc[i, 'color'] = palette[ind]
    source = ColumnDataSource(ColumnDataSource.from_df(m))
    hm = figure(title='Regression Coefficients Summary',
                plot_width=1200, plot_height=400,
                tools=[hover, PanTool(), BoxZoomTool(),
                       WheelZoomTool(), ResetTool(),
                       SaveTool()])
    hm.rect(x='X', y='Y', width=1, height=1,
            fill_color='color', line_color="white", source=source)
    Xlabels = pd.Series(pvals.index, index=np.arange(len(pvals.index)))
    Ylabels = pd.Series(pvals.columns, index=np.arange(len(pvals.columns)), )

    hm.xaxis[0].ticker = FixedTicker(ticks=Xlabels.index)
    hm.xaxis.formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
    """ % Xlabels.to_dict())

    hm.yaxis[0].ticker = FixedTicker(ticks=Ylabels.index)
    hm.yaxis.formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
    """ % Ylabels.to_dict())

    return hm


def _decorate_tree(t, series):
    """ Attaches some default values on the tree for plotting.

    Parameters
    ----------
    t: skbio.TreeNode
        Input tree
    series: pd.Series
        Input pandas series

    """
    for i, n in enumerate(t.postorder()):
        n.size = 30
        if n.is_root():
            n.size = 50
        elif n.name == n.parent.children[0].name:
            n.color = '#00FF00'  # left child is green
        else:
            n.color = '#FF0000'  # right child is red
        if not n.is_tip():
            t.length = series.loc[n.name]
    return t


def _deposit_results(model, output_dir):
    """ Store all of the core regression results into a folder. """
    coefficients = model.coefficients()
    coefficients.T.to_csv(os.path.join(output_dir, 'coefficients.csv'),
                          header=True, index=True)
    residuals = model.residuals()
    residuals.T.to_csv(os.path.join(output_dir, 'residuals.csv'),
                       header=True, index=True)
    predicted = model.predict()
    predicted.T.to_csv(os.path.join(output_dir, 'predicted.csv'),
                       header=True, index=True)
    pvalues = model.pvalues
    pvalues.T.to_csv(os.path.join(output_dir, 'pvalues.csv'),
                     header=True, index=True)

    pvalues = model.pvalues

    def fdr(x):
        return multipletests(x, method='fdr_bh',
                             alpha=0.05 / pvalues.shape[1])[1]

    corrected_pvalues = pvalues.apply(fdr, axis=0)
    corrected_pvalues.T.to_csv(
        os.path.join(output_dir, 'fdr-corrected-pvalues.csv'),
        header=True, index=True)


# OLS summary
def ols_summary(output_dir: str, model: OLSModel,
                tree: TreeNode) -> None:
    """ Summarizes the ordinary least squares fit.

    Parameters
    ----------
    output_dir : str
        Directory where all of the regression results and
        summaries will be stored.
    model : OLSModel
        Ordinary Least Squares model that contains the model fit and the
        regression results.
    tree : TreeNode
        Tree object that defines the partitions of the features. Each of the
        leaves correspond to the balances in the model.
    """
    # Cross validation
    w, h = 500, 300  # plot width and height

    # Explained sum of squares
    ess = model.ess
    # Summary object
    _k, _l = model.kfold(), model.lovo()
    smry = model.summary(_k, _l)
    _deposit_results(model, output_dir)
    t = _decorate_tree(tree, ess)

    p1 = radialplot(t, figsize=(800, 800))
    p1.title.text = 'Explained Sum of Squares'
    p1.title_location = 'above'
    p1.title.align = 'center'
    p1.title.text_font_size = '18pt'

    # 2D scatter plot for prediction on PB
    p2 = _projected_prediction(model, plot_width=w, plot_height=h)
    p3 = _projected_residuals(model, plot_width=w, plot_height=h)
    hm_p = _heatmap_summary(model.pvalues.T, model.coefficients().T)

    # combine the cross validation, explained sum of squares tree and
    # residual plots into a single plot
    p = row(column(p2, p3), p1)
    p = column(hm_p, p)
    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Simplicial Linear Regression Summary</h1>\n')
        index_f.write(smry.as_html())
        index_f.write(
            ('<th>Coefficients</th>\n'
             '<a href="coefficients.csv">'
             'Download as CSV</a><br>\n'
             '<th>Coefficient pvalues</th>\n'
             '<a href="pvalues.csv">'
             'Download as CSV</a><br>\n'
             '<th>FDR corrected coefficient pvalues</th>\n'
             '<a href="fdr-corrected-pvalues.csv">'
             'Download as CSV</a><br>\n'
             '<th>Predicted Balances</th>\n'
             '<a href="predicted.csv">'
             'Download as CSV</a><br>\n'
             '<th>Residuals</th>\n'
             '<a href="residuals.csv">'
             'Download as CSV</a><br>\n')
        )

        plot_html = file_html(p, CDN, 'Diagnostics')
        index_f.write(plot_html)
        index_f.write('</body></html>\n')


# LME summary
def lme_summary(output_dir: str, model: LMEModel, tree: TreeNode) -> None:
    """ Summarizes the ordinary linear mixed effects model.

    Parameters
    ----------
    output_dir : str
        Directory where all of the regression results and
        summaries will be stored.
    model : LMEModel
        Linear Mixed Effects model that contains the model fit and the
        regression results.
    tree : TreeNode
        Tree object that defines the partitions of the features. Each of the
        leaves correspond to the balances in the model.
    """
    # log likelihood
    loglike = pd.Series({r.model.endog_names: r.model.loglike(r.params)
                         for r in model.results})
    w, h = 500, 300  # plot width and height
    # Summary object
    smry = model.summary()

    t = _decorate_tree(tree, -loglike)

    p1 = radialplot(t, figsize=(800, 800))
    p1.title.text = 'Loglikelihood of submodels'
    p1.title_location = 'above'
    p1.title.align = 'center'
    p1.title.text_font_size = '18pt'

    # 2D scatter plot for prediction on PB
    p2 = _projected_prediction(model, plot_width=w, plot_height=h)
    p3 = _projected_residuals(model, plot_width=w, plot_height=h)
    hm_p = _heatmap_summary(model.pvalues.T, model.coefficients().T,
                            plot_width=900, plot_height=400)

    # combine the cross validation, explained sum of squares tree and
    # residual plots into a single plot
    p = row(column(p2, p3), p1)
    p = column(hm_p, p)

    # Deposit all regression results
    _deposit_results(model, output_dir)

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Simplicial Linear Mixed Effects Summary</h1>\n')
        index_f.write(smry.as_html())
        index_f.write(
            ('<th>Coefficients</th>\n'
             '<a href="coefficients.csv">'
             'Download as CSV</a><br>\n'
             '<th>Coefficient pvalues</th>\n'
             '<a href="pvalues.csv">'
             'Download as CSV</a><br>\n'
             '<th>FDR corrected coefficient pvalues</th>\n'
             '<a href="fdr-corrected-pvalues.csv">'
             'Download as CSV</a><br>\n'
             '<th>Predicted Balances</th>\n'
             '<a href="predicted.csv">'
             'Download as CSV</a><br>\n'
             '<th>Residuals</th>\n'
             '<a href="residuals.csv">'
             'Download as CSV</a><br>\n')
        )

        diag_html = file_html(p, CDN, 'Diagnostic plots')
        index_f.write(diag_html)
        index_f.write('</body></html>\n')
