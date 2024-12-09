from gymnasium_env.envs import GridWorldEnv

env = GridWorldEnv(render_mode="human", size=6, max_steps=50, grass_count=3, ou_count=3)

# notes for u guys:
# the game ends if
# - max steps are reached
# - score becomes negative
# - all grass tiles are hit


observation, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
