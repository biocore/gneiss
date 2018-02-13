import numpy as np
import edward as ed
from edward.models import Normal, Poisson, PointMass
import tensorflow as tf

from gneiss.regression._sparse import sparse_matmul, minibatch
from gneiss.balances import sparse_balance_basis
from gneiss.util import sparse_match_tips, rename_internal_nodes
from skbio.stats.composition import _gram_schmidt_basis
from scipy.sparse import coo_matrix
from biom import load_table
from patsy import dmatrix


def poisson_ols(formula, table, metadata, tree=None,
                alpha_mean=0, alpha_scale=1,
                theta_mean=0, theta_scale=1,
                gamma_mean=0, gamma_scale=1,
                beta_mean=0, beta_scale=1,
                iterations=1000, batch_proportion=0.5,
                learning_rate=1e-1,
                beta1=0.9, beta2=0.99,
                batch_size=1000, seed=None,
                check_nans=False,
                summary_dir=None):
    """ Bayesian poisson regression.

    This implementation of Poisson log linear regression attempts to predict
    the  counts from `N` samples and `D` features given a set of `p` measured
    covariates.  A  bifurcating tree can be optionally specified in order to
    perform the ILR transform at the intermediate step. The ILR transform is
    used in order to circumvent the identifiability issues due to drawing
    counts from proportions through the Poisson sampling process.

    The Poisson log linear model has the following model specification

    .. math:
        v_{ij} = \alpha + \theta_i + \gamma_j + g_{i.} \dot \beta_{.j}

        y_{ij} = exp(\Psi v_{ij})

    where :math:y_{ij} represents the counts at sample :math:i and
    feature :math:j. :math:\alpha is the global bias. :math:\theta_i
    is the sample  specific bias, convenient for correcting for total
    count differences across samples. :math:\gamma_i are the feature
    specific biases, which serve as intercepts for each feature.
    :math:\g_{i.} represents the measured covariates for each sample,
    and :math:\beta_{.j} represents the regression coefficients for each
    feature and covariate.  :math:\Psi is the partition matrix used for the
    ILR transform to convert from :math:v_{ij} logits to :math:y_{ij} counts.

    Parameters
    ----------
    formula : str
        Formula representing the statistical equation to be evaluated.
        These strings are similar to how equations are handled in R and
        statsmodels. Note that the dependent variable in this string should
        not be specified, since this method will be run on each of the
        individual balances. See `patsy` for more details.
    table : biom.Tabl
        Contingency table where samples correspond to rows and
        features correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    tree: skbio.TreeNode
        Tree to specify the ILR basis (optional).  If this is not specified,
        then the default gram schmidt basis will be used.
    alpha_mean : float
        Mean of the global scale prior distribution.  This is used to
        correct for global sequencing depth bias. (default: 0)
    alpha_scale : float
        Scale of the global scale prior distribution.  This is used to
        correct for global sequencing depth bias. (default: 1)
    theta_mean : float
        Mean of the sample scale prior distribution.  This is used to
        correct for sample specific sequencing depth sequencing bias.
        (default: 0)
    theta_scale : float
        Scale of the sample scale prior distribution.  This is used to
        correct for sample specific sequencing depth sequencing bias.
        (default: 1)
    gamma_mean : float
        Mean of the feature scale prior distribution.  These are intercept
        terms for each species. (default: 0)
    gamma_scale : float
        Scale of the feature scale prior distribution.  These are intercept
        terms for each species. (default: 1)
    beta_mean : float
        Mean of the coefficient prior distribution.  These are the
        regression coefficients of interest. (default: 0)
    beta_scale : float
        Scale of the coefficient prior distribution.  These are the
        regression coefficients of interest. (default: 1)
    iterations : int
        Number of iterations to perform in the stochastic gradient descent.
        (default: 1000)
    batch_proportion : float
        Proportion of non-zero examples to be selected.
    learning_rate : float
        Learning rate for stochastic gradient optimization.
        (default: 1e-1)
    beta1 : float
        First Momentum rate parameter for ADAM optimizer. (default: 0.9)
    beta2 : float
        Second Momentum rate parameter for ADAM optimizer. (default: 0.99)
    seed : int
        Random seed for Tensorflow. (default: None)
    check_nan : bool
        Specifies if nans should be checked at each step of the tensorflow
        computation graph.  Note that the can drastically slow down
        calculations (default: False)
    summary_dir : str
        Location of diectory to sort summmaries about all of the parameters.
        This can be later visualized in tensorflow.

    Returns
    -------
    PoissonModel
        PoissonModel model object.
    """

    # filters for throwing out empty samples, empty features
    # and zero variance features.

    # lower error bound for variance.  This is the variance for a
    # single individual observed once in 1000 samples.
    eps = 1e-4
    def filter_ids(val_, id_, md_):
        return id_ in metadata.index
    def filter_low_variance(val_, id_, md_):
        return np.var(val_) > eps
    def filter_empty(val_, id_, md_):
        return np.sum(val_) > 0

    metadata = metadata.loc[table.ids(axis='sample')]
    metadata = metadata.reindex(index=table.ids(axis='sample'))
    table = table.filter(filter_ids, axis='sample')
    table = table.filter(filter_empty, axis='sample')
    table = table.filter(filter_empty, axis='observation')
    table = table.filter(filter_low_variance, axis='observation')
    num_features = table.shape[0]

    if tree is None:
        basis = coo_matrix(_gram_schmidt_basis(num_features),
                           dtype=np.float32)
    else:
        tree = rename_internal_nodes(tree)
        table, tree = sparse_match_tips(table, tree)
        basis = sparse_balance_basis(tree)[0]

    # remove the intercept term for now.
    md_data = dmatrix(formula + ' -1 ', metadata, return_type='dataframe')
    sort_f = lambda x: list(md_data.index)
    table = table.sort(sort_f=sort_f, axis='sample')
    model = PoissonOLSModel(
        alpha_mean, alpha_scale,
        theta_mean, theta_scale,
        gamma_mean, gamma_scale,
        beta_mean, beta_scale,
        iterations, batch_proportion,
        learning_rate,
        beta1, beta2,
        batch_size, seed,
        check_nans,
        summary_dir)
    return model.fit(table.matrix_data.tocoo().T,
                     md_data.values, basis.T)

class PoissonOLSModel():
    def __init__(self, alpha_mean=0, alpha_scale=1,
                 theta_mean=0, theta_scale=1,
                 gamma_mean=0, gamma_scale=1,
                 beta_mean=0, beta_scale=1,
                 iterations=1000, batch_proportion=0.5,
                 learning_rate=1e-1,
                 beta1=0.9, beta2=0.99,
                 batch_size=1000, seed=None,
                 check_nans=False,
                 summary_dir=None):
        """ Bayesian Poisson regression model object.

        Parameters
        ----------
        alpha_mean : float
            Mean of the global scale prior distribution.  This is used to
            correct for global sequencing depth bias. (default: 0)
        alpha_scale : float
            Scale of the global scale prior distribution.  This is used to
            correct for global sequencing depth bias. (default: 1)
        theta_mean : float
            Mean of the sample scale prior distribution.  This is used to
            correct for sample specific sequencing depth sequencing bias.
            (default: 0)
        theta_scale : float
            Scale of the sample scale prior distribution.  This is used to
            correct for sample specific sequencing depth sequencing bias.
            (default: 1)
        gamma_mean : float
            Mean of the feature scale prior distribution.  These are intercept
            terms for each species. (default: 0)
        gamma_scale : float
            Scale of the feature scale prior distribution.  These are intercept
            terms for each species. (default: 1)
        beta_mean : float
            Mean of the coefficient prior distribution.  These are the
            regression coefficients of interest. (default: 0)
        beta_scale : float
            Scale of the coefficient prior distribution.  These are the
            regression coefficients of interest. (default: 1)
        iterations : int
            Number of iterations to perform in the stochastic gradient descent.
            (default: 1000)
        batch_proportion : float
            Proportion of non-zero examples to be selected.
        learning_rate : float
            Learning rate for stochastic gradient optimization.
            (default: 1e-1)
        beta1 : float
            First Momentum rate parameter for ADAM optimizer. (default: 0.9)
        beta2 : float
            Second Momentum rate parameter for ADAM optimizer. (default: 0.99)
        seed : int
            Random seed for Tensorflow. (default: None)
        check_nan : bool
            Specifies if nans should be checked at each step of the tensorflow
            computation graph.  Note that the can drastically slow down
            calculations (default: False)
        summary_dir : str
            Location of diectory to sort summmaries about all of the parameters.
            This can be later visualized in tensorflow.
        """
        self.alpha_mean = alpha_mean
        self.alpha_scale = alpha_scale
        self.theta_mean = theta_mean
        self.theta_scale = theta_scale
        self.gamma_mean = gamma_mean
        self.gamma_scale = gamma_scale
        self.beta_mean = beta_mean
        self.beta_scale = beta_scale
        self.iterations = iterations
        self.batch_proportion = batch_proportion
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.seed = seed
        self.check_nans = check_nans
        self.summary_dir = summary_dir


    def fit(self, Y, X, basis=None):
        """  Fits model.

        Parameters
        ----------
        Y : Scipy sparse matrix
            Sparse matrix of observed counts.
        X : np.array
            Design matrix of covariates.
        basis : np.array
            Partition matrix for ILR transform.
            Also called the sequential binary partition.

        Returns
        -------
        self
        """
        N, D = Y.shape
        if basis is None:
            basis = coo_matrix(_gram_schmidt_basis(D), dtype=np.float32)

        p = X.shape[1]   # number of covariates

        # dummy variables for mini-batch size
        batch_idx = tf.placeholder(tf.int32, shape=[self.batch_size, 2],
                                   name='batch_idx')

        # global bias
        alpha = Normal(loc=tf.zeros([]) + self.alpha_mean,
                       scale=tf.ones([]) * self.alpha_scale,
                       name='alpha')
        # sample bias
        theta = Normal(loc=tf.zeros([N, 1]) + self.theta_mean,
                       scale=tf.ones([N, 1]) * self.theta_scale,
                       name='theta')
        # species bias
        gamma = Normal(loc=tf.zeros([1, D-1]) + self.gamma_mean,
                       scale=tf.ones([1, D-1]) * self.gamma_scale,
                       name='gamma')

        # dummy variable for gradient
        G = tf.placeholder(tf.float32, [N, p], name='G')
        # add bias terms for samples
        Gprime = tf.concat([theta, tf.ones([N, 1]), G], axis=1)

        # Specify regression coefficents
        B = Normal(loc=tf.zeros([p, D-1]) + self.beta_mean,
                   scale=tf.ones([p, D-1]) * self.beta_scale,
                   name='B')

        # add bias terms for features
        Bprime = tf.concat([gamma, B], axis=0)

        # Convert basis to SparseTensor
        psi = tf.SparseTensor(
            indices=np.mat([basis.row, basis.col]).transpose(),
            values=basis.data,
            dense_shape=basis.shape)

        # clr transform coefficients first, via psi @ Bprime
        V = tf.transpose(
            tf.sparse_tensor_dense_matmul(psi, tf.transpose(Bprime))
        )
        Vprime = tf.concat([tf.ones([1, D]), V], axis=0)

        # retrieve entries selected by index
        eta = sparse_matmul(
            Gprime, Vprime,
            indices = batch_idx
        )
        # obtain counts
        counts = Poisson( rate=tf.exp(eta + alpha), name='counts' )

        # These are the posterior distributions.
        tf.set_random_seed(self.seed)

        self.qalpha = PointMass(
            params=tf.Variable(tf.random_normal([])) ,
            name='qalpha')

        self.qgamma = PointMass(
            params=tf.Variable(tf.random_normal([1, D-1])) ,
            name='qgamma')

        self.qtheta = PointMass(
            params=tf.Variable(tf.random_normal([N, 1])),
            name='qtheta')

        self.qB = PointMass(
            params=tf.Variable(tf.random_normal([p, D-1])) ,
            name='qB')

        # a placeholder for the microbial counts
        # since we will be manually feeding it into the inference
        # via minibatch SGD
        counts_ph = tf.placeholder(tf.float32, shape=[self.batch_size],
                                   name='counts_placeholder')

        self.inference = ed.MAP({
            theta: self.qtheta,
            alpha: self.qalpha,
            gamma: self.qgamma,
            B: self.qB},
            data={G: X, counts: counts_ph}
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=self.beta1,
                                           beta2=self.beta2)

        # adds checks for nans
        if self.check_nans:
            tf.add_check_numerics_ops()

        saver = tf.train.Saver()
        if self.summary_dir is not None:
            self.inference.initialize(
                n_iter=self.iterations,
                optimizer=optimizer,
                n_print=100,
                log_vars=[
                    self.qB,
                    self.qtheta,
                    self.qgamma,
                    self.qalpha
                ],
                logdir=self.summary_dir
            )
        else:
            self.inference.initialize(
                n_iter=self.iterations,
                optimizer=optimizer,
                n_print=100
            )
        # initialize all tensorflow variables
        tf.global_variables_initializer().run()
        for i in range(self.inference.n_iter):
            # get batches
            idx, idx_data = minibatch(self.batch_size, Y,
                                      self.batch_proportion)

            info_dict = self.inference.update(
                feed_dict={batch_idx: idx, counts_ph: idx_data})
            self.inference.print_progress(info_dict)

        if self.summary_dir is not None:
            tf.summary.tensor_summary('qB', self.qB)
            tf.summary.tensor_summary('theta', self.qtheta)
            tf.summary.scalar('beta_mean', self.beta_mean)
            tf.summary.scalar('beta_scale', self.beta_scale)
            tf.summary.scalar('theta_mean', self.theta_mean)
            tf.summary.scalar('theta_scale', self.theta_scale)
            tf.summary.scalar('gamma_mean', self.gamma_mean)
            tf.summary.scalar('gamma_scale', self.gamma_scale)
            tf.summary.scalar('iteration', self.iterations)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('batch_size', self.batch_size)
            tf.summary.scalar('beta1', self.beta1)
            tf.summary.scalar('beta2', self.beta2)
        return self

    def update(self, feed_dict):
        """ Update inference with additional data."""
        raise NotImplemented()

