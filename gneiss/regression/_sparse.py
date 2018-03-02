"""
Contains functions required for sparse learning.
"""
import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state


def sparse_matmul(A, B, indices):
    """ Sparse matrix multiplication.

    This will try to evaluate the following product

    A[i] @ B[j]

    where i, j are the row and column indices specified in `indices`.

    Parameters
    ----------
    A : tf.Tensor
       Left 2D tensor
    B : tf.Tensor
       Right 2D tensor
    indices : tf.Tensor
       2D tensor of indices to be evaluated.  The first column consists
       of row indices, and the second column consists of column indices.

    Returns
    -------
    tf.Tensor
       Result stored in a sparse tensor format, where the values
       are derived from A[i] @ B[j] where i, j are the row and column
       indices specified in `indices`.
    """
    row_index = tf.gather(indices, 0, axis=0)
    col_index = tf.gather(indices, 1, axis=0)
    A_flat = tf.gather(A, row_index, axis=0)
    B_flat = tf.transpose(tf.gather(B, col_index, axis=1))
    values = tf.reduce_sum(tf.multiply(A_flat, B_flat), axis=1)
    return values


def minibatch(M, Y, p, seed=0):
    """ Get's minibatch data

    This performs a minibatch bootstrap, randomly selecting
    entrices from the sparse matrix `Y`.

    Parameters
    ----------
    M : int
        batch size
    Y : scipy.sparse.coo_matrix
        Scipy sparse matrix in COO-format.
    p : float
        Portion of nonzero entries to be sampled.
    seed : int or np.random.RandomState
        Random seed
    Returns
    -------
    batch_index : np.array
        Selected rows and columns in th sparse matrix Y.
    batch_data : np.array
        Selected data
    """
    state = check_random_state(seed)
    nonzeroM = round(M * p)
    y_data = Y.data
    y_row = Y.row
    y_col = Y.col
    # get positive sample
    positive_idx = state.choice(len(y_data), nonzeroM)
    positive_row = y_row[positive_idx]
    positive_col = y_col[positive_idx]
    positive_data = y_data[positive_idx]

    # store all of the positive (i, j) coords
    idx = np.vstack((y_row, y_col)).T
    idx = set(map(tuple, idx.tolist()))

    # get negative sample
    N, D = Y.shape
    zeroM = round(M * (1-p))
    negative_row = np.zeros(zeroM)
    negative_col = np.zeros(zeroM)
    negative_data = np.zeros(zeroM)
    for k in range(zeroM):
        i, j = state.randint(N), state.randint(D)
        while (i, j) in idx:
            i, j = state.randint(N), state.randint(D)
        negative_row[k] = i
        negative_col[k] = j
    batch_row = np.hstack((positive_row, negative_row))
    batch_col = np.hstack((positive_col, negative_col))
    batch_data = np.hstack((positive_data, negative_data))
    batch_idx = np.vstack((batch_row, batch_col)).T
    return batch_idx, batch_data
