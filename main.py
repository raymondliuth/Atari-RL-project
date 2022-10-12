import gym
env = gym.make('AirRaid-v4', render_mode = 'human')
env.reset()
done = False
while not done:
    env.step(env.action_space.sample())
env.close()