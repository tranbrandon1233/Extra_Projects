import tensorflow as tf
tf.random.set_seed(0)

mask = tf.constant([[True, False, False, True, False],
[False, True, True, False, False],
[False, False, False, True, False]]) # Example mask

# Dimensions (N, L) where N = batch size, L = length of mask
N, L = mask.shape
# Create a range tensor (0, 1, ..., L - 1) for each row of the mask
indices = tf.tile(tf.range(L)[tf.newaxis, :], [N, 1]) #tf.expand_dims(tf.range(L), axis=0), 0)
# Mask the indices tensor using the mask to get only indices corresponding to True elements
indices = tf.where(mask, indices, -1)
# Get the last index for each row (maximum of each row/argmax)
indices = tf.reduce_max(indices,axis=1)
rows = tf.range(N) # (0, 1, ..., N - 1)
# Create [i,j] pairs to use tf.scatter_nd_update
new_mask_indices = tf.stack([rows, indices], axis=1)
new_mask = tf.scatter_nd(new_mask_indices, updates=tf.ones(N, dtype=tf.bool), shape=mask.shape)
print(new_mask)
def find_last_true(mask):
    placeholder = tf.zeros_like(mask, dtype=tf.bool)

    # get the indices of last True in each row
    reverse_mask = tf.reverse(mask, axis=[1])
    exclusive_sum = tf.cumsum(tf.cast(reverse_mask, tf.float32), axis=-1, reverse=True, exclusive=True)
    index = tf.cast(tf.map_fn(fn=lambda x: tf.math.argmin(x, output_type=tf.int32), elems=exclusive_sum), tf.int32)
    # create a new mask with a single True per row at the found index
    brow_idx = tf.range(tf.shape(mask)[0])
    gather_idces = tf.stack([brow_idx, index], axis=-1)
    values = tf.ones_like(index, dtype=tf.bool)
    new_mask = tf.scatter_nd(indices=gather_idces, updates=values, shape=tf.shape(mask))
    return new_mask
new_mask = find_last_true(mask)
print(new_mask) # Output: [[False, False, False, True, F