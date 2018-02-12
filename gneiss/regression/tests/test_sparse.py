import unittest
import numpy as np
import numpy.testing as npt
import tensorflow as tf
from gneiss.regression._sparse import sparse_matmul, minibatch
from scipy.sparse import coo_matrix


class TestSparseMatMul(tf.test.TestCase):

    def test_sparse_matmul(self):

        A = np.array(
            [[1, 5, 1, 6],
             [1, 3, 1, 6],
             [3, 5, 9, 5],
             [1, 5, 1, 10],
             [2, 1, 2, 6]]
        )

        B = np.array(
            [[1, 5],
             [1, 3],
             [3, 5],
             [1, 5]]
        )

        indices = tf.convert_to_tensor(
            np.array([[1, 3, 4], [1, 0, 1]]))
        exp_idx = np.array([[1, 2],
                            [3, 3],
                            [4, 3]])
        exp_data = np.array([A[1] @ B[:, 1],
                             A[3] @ B[:, 0],
                             A[4] @ B[:, 1]])
        with self.test_session():
            tA = tf.convert_to_tensor(A)
            tB = tf.convert_to_tensor(B)
            res = sparse_matmul(tA, tB, indices)
            exp = tf.convert_to_tensor(exp_data, name='Sum')
            npt.assert_allclose(res.eval(), exp.eval())


class TestMinibatch(unittest.TestCase):
    def test_minibatch(self):
        A = np.array([[1, 0, 0, 10, 0, 1],
                      [0, 0, 10, 0, 1, 2],
                      [0, 10, 0, 1, 5, 0],
                      [0, 0, 0, 10, 0, 1],
                      [0, 2, 0, 0, 0, 1],
                      [1, 0, 0, 5, 0, 0],
                      [1, 0, 0, 10, 0, 1],
                      [0, 5, 0, 0, 0, 1],
                      [1, 0, 0, 10, 0, 0]])
        mat = coo_matrix(A)
        res = minibatch(10, mat, p=0.5, seed=0)
        res_idx = res[0].astype(np.int)
        res_data = res[1]
        exp_idx = np.array(
            [[4, 5],
             [6, 0],
             [8, 3],
             [0, 0],
             [1, 2],
             [3, 1],
             [7, 0],
             [8, 4],
             [1, 0],
             [8, 1]]
        )
        exp_data = np.array([1., 1., 10., 1., 10.,
                             0., 0., 0., 0., 0.])
        npt.assert_allclose(exp_idx, res_idx)
        npt.assert_allclose(exp_data, res_data)


if __name__ == "__main__":
    unittest.main()
