import argparse
from gymnasium_env.envs import GridWorldEnv
from model.train import train_model
from model.evaluation import evaluate

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train, evaluate, or run both for the PPO GridWorld model.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate", "both"], 
        default="both", 
        help="Choose to train, evaluate, or do both."
    )
    args = parser.parse_args()

    # Define paths and configurations
    model_path = "policy/ppo_gridworld_model"
    env_config_train = {
        "size": 5,
        "max_steps": 5000,
        "grass_count": 3,
        "ou_count": 5,
        "penalty_scaling": 0.1
    }
    env_config_evaluate = {
        "render_mode": "human", 
        "size": 5,
        "max_steps": 5000,
        "grass_count": 3,
        "ou_count": 5,
    }

    # Run based on the selected mode
    if args.mode in ["train", "both"]:
        print("Starting training...")
        train_model(model_path=model_path, total_timesteps=200000, env_config=env_config_train)

    if args.mode in ["evaluate", "both"]:
        print("Starting evaluation...")
        evaluate(model_path=model_path, env_config=env_config_evaluate, max_episodes=5)
