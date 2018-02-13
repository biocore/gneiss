import unittest
import edward as ed
import numpy as np
import numpy.testing as npt
import tensorflow as tf
from scipy.sparse import coo_matrix
from gneiss.regression._sparse import sparse_matmul, minibatch
from gneiss.regression._poisson import poisson_ols, PoissonOLSModel
from skbio.stats.composition import _gram_schmidt_basis
from gneiss.sim.generators import band_table


class TestPoissonOLS(tf.test.TestCase):

    def setUp(self):
        # Build basis for the ilr transform.
        num_samples = 10
        num_features = 5

        basis = coo_matrix(_gram_schmidt_basis(num_features),
                           dtype=np.float32).T
        res = band_table(num_samples, num_features, tree=None,
                         mu=5, sigma=2, low=2, high=8,
                         spread=1, feature_bias=1, alpha=7, seed=0)
        self.table = res[0]
        self.metadata = res[1]


    def test_poisson_regression(self):
        model = poisson_ols('G', self.table, self.metadata,
                            batch_size=12, seed=0)

        exp_beta = np.array([[-0.42114484, -0.14152822, 0.8636342, 0.6733759 ]],
                            dtype=np.float32)
        exp_theta = np.array([2.4314125e+00, -9.6548976e-23, -1.2869763e-01,
                              -3.2601155e-02, -8.9476071e-04, 2.4937815e-04,
                              -1.1222001e-01, -2.0178803e-03, -2.2661078e-01,
                              -1.6405735e+00],
                             dtype=np.float32).reshape(-1, 1)
        exp_gamma = np.array([[3.3982987, 1.5410572,
                               2.0590074, 1.5908878]],
                             dtype=np.float32)
        exp_alpha = 0.4301655

        sess = ed.get_session()
        res_beta = model.qB.eval(session=sess)
        res_theta = model.qtheta.eval(session=sess)
        res_gamma = model.qgamma.eval(session=sess)
        res_alpha = model.qalpha.eval(session=sess)
        npt.assert_allclose(res_beta, exp_beta)
        npt.assert_allclose(res_theta, exp_theta)
        npt.assert_allclose(res_gamma, exp_gamma)
        npt.assert_allclose(res_alpha, exp_alpha)


if __name__ == "__main__":
    unittest.main()

