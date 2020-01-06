import tensorflow as tf
import keras
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

"""### Custom loss triplet loss"""

def _pairwise_distances(embeddings, batch_size=256, squared=False):
    """
    Compute 2D matrix of distances between all the embeddings

    Args:
        * embeddings: tensor with shape (batch_size, embed_dim)
        * squared: Boolean. True: distance is pairwise squared euclidean distance
                            False: distance is pairwise euclidean distance
    """

    # product matrix, shape (batch_size, batch_size)
    dot_product = math_ops.matmul(embeddings, array_ops.transpose(embeddings))

    diag = K.eye(int(batch_size))

    #make it 3D by adding a dummy batch dimension
    diag = K.expand_dims(diag,0)

    #annulate the diagonal from the input
    noDiagInput = diag * dot_product

    square_norm = K.reshape(noDiagInput[noDiagInput>0], (int(batch_size), ))

    distances = math_ops.add(
        K.expand_dims(square_norm, 1),
        K.expand_dims(square_norm, 0) - 2.0 * dot_product
    )

    distances = K.maximum(distances, 0.0)

    if not squared:
        mask = K.cast(K.equal(distances, 0.0), dtype=float)
        distances = distances + mask * 1e-16

        distances = K.sqrt(distances)
        distances = distances * (1.0 - mask)
    
    return distances

def _get_triplet_mask(labels, batch_size=256):
    """
    Return a 3D mask where mask[a, p, n] is True and the triplet (a, p, n) is valid

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = K.cast(K.eye(int(batch_size)), dtype=bool)
    indices_not_equal = math_ops.logical_not(indices_equal)

    i_not_equal_j = K.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.expand_dims(indices_not_equal, 0)

    distinct_indices = math_ops.logical_and(math_ops.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))
    i_equal_j = K.expand_dims(label_equal, 2)
    i_equal_k = K.expand_dims(label_equal, 1)

    valid_labels = math_ops.logical_and(i_equal_j, math_ops.logical_not(i_equal_k))
    # Combine the two masks
    mask = math_ops.logical_and(distinct_indices, valid_labels)

    return mask

def batch_all_triplet_loss(labels, y_pred):
    # del labels
    margin = 1.
    labels = labels
    squared = False
    batch_size = 5

    labels = tf.cast(labels, dtype='int32')
    embeddings = y_pred

    pairwise_dist = _pairwise_distances(embeddings, batch_size, squared=squared)
    
    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels, batch_size)

    mask = K.cast(mask, dtype=float)
    triplet_loss = math_ops.mul(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = K.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = K.cast(K.greater(triplet_loss, 1e-16), dtype=float)

    num_positive_triplets = math_ops.reduce_sum(valid_triplets)
    num_valid_triplets = math_ops.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = math_ops.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    
    return triplet_loss
