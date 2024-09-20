# UTILITY FUNCTIONS
#####################
import tensorflow as tf


def tf_mat_vec_dot(matrix, vector):
    """
    Matrix product between matrix and vector.
    """
    return tf.matmul(matrix, tf.expand_dims(vector, 1))[:, 0]  # TODO: tf.linalg.matvec(matrix, vector)


def tf_outer_product(first_vec, second_vec):
    """
    Outer product of two vectors, outer(v, j)_ij = v[i] * v[j].
    Source: https://www.tensorflow.org/api_docs/python/tf/einsum
    """
    return tf.einsum('i,j->ij', first_vec, second_vec)


def check_nan(x):
    """
    Utility function to avoid feeding NaNs into the network.
    """
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)



