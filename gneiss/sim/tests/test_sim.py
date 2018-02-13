import numpy as np
import numpy.testing as npt
import unittest
from gneiss.sim._sim import (
    chain_interactions, linregress
)
from sklearn.utils import check_random_state


class TestSim(unittest.TestCase):

    def test_chain_interactions(self):

        gradient = np.linspace(0, 10, 5)
        mu = np.array([4, 6])
        sigma = np.array([1, 1])

        exp = np.array([[1.33830226e-04, 6.07588285e-09],
                        [1.29517596e-01, 8.72682695e-04],
                        [2.41970725e-01, 2.41970725e-01],
                        [8.72682695e-04, 1.29517596e-01],
                        [6.07588285e-09, 1.33830226e-04]])
        res = chain_interactions(gradient, mu, sigma)
        npt.assert_allclose(exp, res)

    def test_linregress(self):
        y = np.array([[1, 2, 3, 4, 5],
                      [2, 4, 6, 8, 10]]).T
        x = np.array([[1, 1, 1, 1, 1],
                      [1, 2, 3, 4, 5]]).T
        exp_beta = np.array([[0, 0],
                             [1, 2]])
        py, _, res_beta = linregress(y, x)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-7, rtol=1e-4)
        npt.assert_allclose(y, py)


if __name__ == "__main__":
    unittest.main()
