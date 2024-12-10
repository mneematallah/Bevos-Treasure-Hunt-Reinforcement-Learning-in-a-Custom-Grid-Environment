from gymnasium_env.envs import GridWorldEnv
from model.train import train_model
from model.evaluation import evaluate


if __name__ == "__main__":
    # env = GridWorldEnv(render_mode="human", size=5, max_steps=5000, grass_count=3, ou_count=5)
    # Load/Intialize policy and environment
    model_path = "policy/ppo_gridworld_model"
    env_config = {
        "size": 5,
        "max_steps": 5000,
        "grass_count": 3,
        "ou_count": 5,
    }

    # Train the model with CUDA
    # print("Starting training...")
    # train_model(model_path=model_path, total_timesteps=200000, env_config=env_config)

    # Evaluate the model with CUDA
    print("Starting evaluation...")
    evaluate(model_path=model_path, env_config=env_config, max_episodes=50)
