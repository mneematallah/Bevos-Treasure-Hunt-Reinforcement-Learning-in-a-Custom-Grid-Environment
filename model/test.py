# """
# Script to test if environment is correctly initialized/ compatible with PPO model used for training agent
# """

# from stable_baselines3.common.vec_env import DummyVecEnv
# from gymnasium_env.envs import GridWorldEnv

# def make_env():
#     return GridWorldEnv(render_mode=None, size=5, max_steps=5000, grass_count=3, ou_count=5)

# # Wrap the environment
# vec_env = DummyVecEnv([make_env])

# # Test the vectorized environment
# obs = vec_env.reset()
# print(f"Initial Observation: {obs}")

# # Take a random step
# action = [vec_env.action_space.sample()]  # Single environment in VecEnv still expects a list
# obs, rewards, dones, infos = vec_env.step(action)
# print(f"Next Observation: {obs}")
# print(f"Rewards: {rewards}")
# print(f"Dones: {dones}")

import pygame
pygame.init()
screen = pygame.display.set_mode((512, 512))
pygame.display.set_caption("Test Pygame Window")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255, 255, 255))  # White background
    pygame.display.update()
pygame.quit()