import time

import tensorflow as tf
import keras
import numpy as np

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

q_model = tf.keras.models.load_model("attention_model/30000_checkpoint")

env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=False, scale=True)




for i in range(50):
    state = np.array(env.reset())
    done = False
    while not done:
        env.render()  # ; Adding this line would show the attempts
        time.sleep(0.01)
        # of the agent in a pop up window.
        # action = np.random.choice(4)

        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = q_model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

        state_next, reward, done, _ = env.step(action)
        print(done)
        state_next = np.array(state_next)