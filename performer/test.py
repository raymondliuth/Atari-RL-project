import tensorflow as tf
from keras import layers
import keras

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from patches import Patches
from PatchEncoder import PatchEncoder
import numpy as np
from performer.fast_attention import SelfAttention

s = SelfAttention(128, 64, 0.2)

inputs = layers.Input(shape=(84, 84))

def model():
    inputs = layers.Input(shape=(84, 84))
    resized_image = tf.expand_dims(tf.convert_to_tensor(inputs), axis=3)

    return keras.Model(inputs=inputs, outputs=resized_image)


env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=False, scale=True)

state = np.array(env.reset())

m = model()
print(m(state))
