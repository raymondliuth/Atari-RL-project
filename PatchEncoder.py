import tensorflow as tf
from keras import layers


class PatchEncoder(layers.Layer):
    """
    input patch: takes in one batch of patch [number of patches, patch dims (patch size * patchsize)]
    output: Projected dimension patch [number of patches, projected dims]
    """
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded