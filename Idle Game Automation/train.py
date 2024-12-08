import os
import time
import datetime
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from src.environment import IdleGameEnvironment
import keyboard
import logging
import torch

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Saving checkpoints for every hour of training
class HourlyCheckpoint(BaseCallback):
    def __init__(self, save_path, save_freq=3600, verbose=0):
        super(HourlyCheckpoint, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.last_save_time = time.time()
        self.training_start_time = time.time()
        self.rewards = []
        self.scores = []
        self.actions = []
        self.q_values = []
        self.training_active = True
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Force saving with keyboard interruption
        if keyboard.is_pressed('ctrl+shift+s'):
            logging.info("Force saving the model and stopping training...")
            self.save_metrics()
            return False

        # Capture metrics from the most recent step
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        action = self.locals['actions'][0]
        score = info.get('score', 0)
        
        self.rewards.append(reward)
        self.scores.append(score)
        self.actions.append(action)

        # Currently ignored due to q_values not reflecting properly in metrics. Requires better implementation
        obs = self.locals['new_obs'][0]
        with torch.no_grad():
            q_values = self.model.q_net(self.model.q_net.obs_to_tensor(obs)[0])
        self.q_values.append(q_values.cpu().numpy())

        if self.step_count % 100 == 0:
            logging.info(f"Step {self.step_count}:")
            logging.info(f"  Reward = {reward:.2f}")
            logging.info(f"  Score = {score:.2f}")
            logging.info(f"  Action = {action}")
            logging.info(f"  Q-values = {q_values.cpu().numpy()}")

        current_time = time.time()
        if current_time - self.last_save_time >= self.save_freq:
            self.save_metrics()
            self.last_save_time = current_time

        return True

    def save_metrics(self):
        hours_passed = int((time.time() - self.training_start_time) / 3600)
        model_path = f"{self.save_path}/model_{hours_passed}h"
        self.model.save(model_path)
        
        metrics_path = f"{self.save_path}/metrics_{hours_passed}h.npz"
        np.savez(metrics_path, 
                 rewards=np.array(self.rewards), 
                 scores=np.array(self.scores), 
                 actions=np.array(self.actions),
                 q_values=np.array(self.q_values),
                 training_time=time.time() - self.training_start_time)
        
        logging.info(f"Saved model and metrics at {hours_passed} hours of training")
        logging.info(f"Current metrics - Rewards: {len(self.rewards)}, Scores: {len(self.scores)}, Actions: {len(self.actions)}")
        logging.info(f"Average Reward: {np.mean(self.rewards):.2f}, Average Score: {np.mean(self.scores):.2f}")
        logging.info(f"Action distribution: {np.unique(self.actions, return_counts=True)}")

def train_dqn(env, save_path, total_timesteps=10000000, buffer_size=100000, learning_rate=0.001):
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'training.log')
    setup_logging(log_file)
    
    callback = HourlyCheckpoint(save_path)
    
    model = DQN("MlpPolicy", env, verbose=1, buffer_size=buffer_size, learning_rate=learning_rate)
    logging.info("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=100)
    logging.info("Training completed.")
    
    return model, callback

def main():
    # Update the vision model path here
    vision_model_path = "vision_models/FinalModel198Epochs.pt"
    
    logging.info("Initialising environment...")
    env = IdleGameEnvironment(vision_model_path, "IdleGame")
    
    # Saving the trained model to be used later
    save_path = os.path.join("trained_agents", f"agent_{datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')}")
    logging.info(f"Training model. Saving to {save_path}")
    model, callback = train_dqn(env, save_path=save_path)
    
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    final_metrics_path = os.path.join(save_path, "final_metrics.npz")
    np.savez(final_metrics_path, 
             rewards=np.array(callback.rewards), 
             scores=np.array(callback.scores), 
             actions=np.array(callback.actions),
             q_values=np.array(callback.q_values),
             training_time=time.time() - callback.training_start_time)
    
    logging.info("Final metrics saved!")
    logging.info(f"Total steps: {callback.step_count}")
    logging.info(f"Total training time: {time.time() - callback.training_start_time:.2f} seconds")
    logging.info(f"Average Reward: {np.mean(callback.rewards):.2f}")
    logging.info(f"Average Score: {np.mean(callback.scores):.2f}")
    logging.info(f"Action distribution: {np.unique(callback.actions, return_counts=True)}")

if __name__ == "__main__":
    main()