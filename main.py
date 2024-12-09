from gymnasium_env.envs import GridWorldEnv

env = GridWorldEnv(render_mode="human", size=5)

observation, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
