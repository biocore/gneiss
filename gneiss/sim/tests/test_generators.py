import unittest
from bayesian_regression.util.generators import band_table, block_table

from biom import Table
import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


class TestGenerator(unittest.TestCase):

    def setUp(self):
        pass

    def test_band_table(self):
        res = band_table(5, 6, alpha=3)
        res_table, res_md, res_fmd, res_beta, res_theta, res_gamma = res
        mat = np.array(
            [[17.0, 17.0, 4.0, 1.0, 0.0],
             [0.0, 1.0, 3.0, 4.0, 1.0],
             [1.0, 1.0, 5.0, 10.0, 4.0],
             [1.0, 0.0, 1.0, 5.0, 8.0],
             [0.0, 0.0, 5.0, 3.0, 6.0],
             [0.0, 0.0, 2.0, 5.0, 4.0]]
        )

        samp_ids = ['S0', 'S1',	'S2', 'S3', 'S4']
        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']
        exp_fmd = pd.DataFrame({'mu': [3.045444, 5.800314, 6.957476,
                                       8.528105, 8.735116, 9.481786]},
                               index=feat_ids)
        exp_table = Table(mat, feat_ids, samp_ids)
        exp_md = pd.DataFrame({'G': [2., 4., 6., 8., 10.]},
                              index=samp_ids)

        exp_beta = np.array(
            [[-0.48699685, -0.51737237, -0.70588861, -0.59306809, -0.6546415 ]]
        )
        exp_theta = np.array([-4.764716, -2.908262, -1.886161,
                              -2.048065, -2.695843])
        exp_gamma = np.array(
            [[3.1040167, 2.59893833, 4.4744781, 4.35601705, 4.91757565]]
        )

        self.assertEqual(exp_table, res_table)
        pdt.assert_frame_equal(exp_md, res_md)
        pdt.assert_frame_equal(exp_fmd, res_fmd)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-6)
        npt.assert_allclose(exp_theta, res_theta, atol=1e-6)

    def test_block_table(self):
        res = block_table(10, 8, sigma=1,
                  mu_num=7, mu_null=5, mu_denom=3,
                  spread=1, low=3, high=7)
        res_table, res_md, res_fmd, res_beta, res_theta, res_gamma = res
        mat = np.array([[0., 0., 0., 0., 1., 0., 0., 2., 5., 1.],
                        [0., 1., 0., 3., 229., 0., 5., 269., 0., 385.],
                        [0., 1., 1., 7., 93., 0., 38., 95., 4., 8.],
                        [276., 45., 315., 101., 2., 354., 164., 4., 125., 0.],
                        [2., 1., 4., 4., 3., 31., 134., 3., 227., 0.],
                        [17., 28., 35., 145., 31., 2., 14., 2., 3., 0.],
                        [83., 305., 19., 99., 43., 0., 13., 1., 0., 0.],
                        [11., 18., 36., 77., 6., 47., 82., 2., 33., 0.]])


        samp_ids = ['S0', 'S1',	'S2', 'S3', 'S4',
                    'S5', 'S6', 'S7', 'S8', 'S9']
        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

        exp_fmd = pd.DataFrame(
            {'class': np.array([1, 1, 0, 0, 0, 0, -1, -1]),
             'mu': np.array([8.76405235, 7.40015721, 6.86755799,
                             4.02272212, 5.95008842, 4.84864279,
                             3.97873798, 5.2408932])},
            index=feat_ids)

        exp_table = Table(mat, feat_ids, samp_ids)
        exp_md = pd.DataFrame({'G': [0., 0., 0., 0., 0.,
                                     1., 1., 1., 1., 1.]},
                              index=samp_ids)
        exp_beta = np.array([[0.9644195 , 0.9916733 , 3.16491905,
                              0.72764693, 1.59959892,
                              2.15728319, 0.68762395]])
        exp_theta = np.array([-6.34614442, -5.0345698 , -5.64265615,
                              -3.5113818 , -3.51443105, -5.58177084,
                              -3.09031576, -3.86948337, -3.71539897,
                              -8.40406143])
        exp_gamma = np.array([[-7.8977583 , -7.19184908, -18.64689094,
                               -4.50511347, -9.53372303, -12.1336633 ,
                               -4.72704718]])

        self.assertEqual(exp_table, res_table)
        pdt.assert_frame_equal(exp_md, res_md)
        pdt.assert_frame_equal(exp_fmd, res_fmd)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-6)
        npt.assert_allclose(exp_theta, res_theta, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
