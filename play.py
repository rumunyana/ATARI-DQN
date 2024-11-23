import gymnasium as gym
from stable_baselines3 import DQN
import ale_py

def play_breakout():
    # Register ALE environments with Gymnasium
    gym.register_envs(ale_py)
    
    model = DQN.load("policy")
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                print(f"Episode {episode + 1} reward: {total_reward}")
                break
    
    env.close()

if __name__ == "__main__":
    play_breakout()
