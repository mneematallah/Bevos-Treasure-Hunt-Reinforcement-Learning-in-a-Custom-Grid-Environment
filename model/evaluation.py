from stable_baselines3 import PPO
from gymnasium_env.envs import GridWorldEnv
import torch


def evaluate(model_path, env_config=None, max_episodes=5):
    """
    Runs a trained PPO model on the GridWorldEnv and visualizes its behavior.

    :param model_path: str - File path to the saved model.
    :param env_config: dict - A dictionary of environment parameters (optional).
    :param max_episodes: int - Number of episodes to run for evaluation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Default environment configuration
    if env_config is None:
        env_config = {
            "render_mode": "human",  # Enable rendering for visualization
            "size": 5,
            "max_steps": 5000,
            "grass_count": 3,
            "ou_count": 5,
        }

    # Create the environment
    env = GridWorldEnv(**env_config)

    # Load the trained model
    model = PPO.load(model_path, device=device)
    print(f"Loaded model from {model_path}")

    try:
        for episode in range(max_episodes):
            print(f"Starting Episode {episode + 1}")
            observation, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Get the best action from the model
                action, _ = model.predict(observation, deterministic=True)
                # Take the action in the environment
                observation, reward, done, truncated, info = env.step(action)
                # Accumulate rewards
                total_reward += reward
                # Render the environment
                env.render()

            print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    finally:
        # Ensure the environment is closed properly
        env.close()
