import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import ale_py
import numpy as np

gym.register_envs(ale_py)

def wrap_breakout_env():
    """Create and wrap Breakout environment with proper preprocessing"""
    env = gym.make("ALE/Breakout-v5")
    env = Monitor(env)
    return DummyVecEnv([lambda: env])

def setup_training():
    """Configure and return the DQN agent"""
    env = wrap_breakout_env()
    
    agent = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./breakout_tensorboard/"
    )
    
    return agent, env

def train_agent():
    """Train the DQN agent on Breakout"""
    agent, env = setup_training()
    
    eval_env = wrap_breakout_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    try:
        agent.learn(
            total_timesteps=50000,
            callback=eval_callback,
            progress_bar=True
        )
        agent.save("policy")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train_agent()