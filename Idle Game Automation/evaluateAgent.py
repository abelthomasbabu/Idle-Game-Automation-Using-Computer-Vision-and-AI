import os
import torch
import numpy as np
from stable_baselines3 import DQN
from src.environment import IdleGameEnvironment
import matplotlib.pyplot as plt
from datetime import datetime
import keyboard
import time
from collections import Counter
import logging

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def evaluate_model(model, env, total_steps=100000, log_interval=100):
    rewards = []
    scores = []
    actions = []
    q_values = []
    cumulative_reward = 0
    action_counts = Counter()
    start_time = time.time()

    obs, _ = env.reset()
    
    logging.info("Starting Evaluation")
    
    for step in range(total_steps):
        if keyboard.is_pressed('ctrl+shift+s'):
            logging.warning("Force save triggered. Stopping evaluation...")
            break

        # Let model choose the action
        action = model.predict(obs, deterministic=True)
        action_key = action.item() if isinstance(action, np.ndarray) else action
        obs, reward, _, info = env.step(action)
        
        cumulative_reward += reward
        current_score = info.get('score', 0)
        action_counts[action_key] += 1
        
        rewards.append(reward)
        scores.append(current_score)
        actions.append(action_key)
        
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            q_value = model.q_net(obs_tensor)
        q_values.append(q_value.cpu().numpy())
        
        if step % log_interval == 0:
            time_elapsed = time.time() - start_time
            
            logging.info(f"Step {step}:")
            logging.info(f"  Cumulative Reward = {cumulative_reward:.2f}")
            logging.info(f"  Score = {current_score:.2f}")
            logging.info(f"  Actions taken: {dict(action_counts)}")
            logging.info(f"  Time taken: {time_elapsed:.2f} seconds")
            logging.info(f"  Q-values: {q_value.cpu().numpy()}")
            
            # Reset action counts for the next interval
            action_counts.clear()
        
        if not np.isfinite(reward):
            logging.error(f"Non-finite reward detected at step {step}: {reward}")

    logging.info(f"Evaluation completed after {step+1} steps")
    return rewards, scores, actions, q_values, time.time() - start_time

def plot_metrics(rewards, scores, actions, q_values, save_path):
    # Plotting rewards and scores
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(np.cumsum(rewards))
    ax1.set_title('Cumulative Reward over Time')
    ax1.set_xlabel('Evaluation Step')
    ax1.set_ylabel('Cumulative Reward')

    ax2.plot(scores)
    ax2.set_title('Score over Time')
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reward_score_metrics.png'))
    plt.close()

    # Action distribution and Q-Values require better implementation to reflect the training and evaluation process better
    # Plotting action distribution 
    plt.figure(figsize=(10, 6))
    action_counts = Counter(actions)
    plt.bar(action_counts.keys(), action_counts.values())
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_path, 'action_distribution.png'))
    plt.close()

    # Plotting Q-values
    plt.figure(figsize=(10, 6))
    q_values_array = np.array(q_values).squeeze()
    for i in range(q_values_array.shape[1]):
        plt.plot(q_values_array[:, i], label=f'Action {i}')
    plt.title('Q-values over Time')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Q-value')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'q_values.png'))
    plt.close()

def save_results(eval_folder, rewards, scores, actions, q_values, eval_time):
    summary_path = os.path.join(eval_folder, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"Total Steps: {len(rewards)}\n")
        f.write(f"Final Cumulative Reward: {np.sum(rewards):.2f}\n")
        f.write(f"Final Score: {scores[-1]:.2f}\n")
        f.write(f"Average Reward per step: {np.mean(rewards):.2f}\n")
        f.write(f"Score Increase: {scores[-1] - scores[0]:.2f}\n")
        f.write(f"Total Evaluation Time: {eval_time:.2f} seconds\n")
    
    logging.info(f"Evaluation summary saved to {summary_path}")
    
    plot_metrics(rewards, scores, actions, q_values, eval_folder)
    logging.info(f"Metrics plots saved in {eval_folder}")
    
    np.savez(os.path.join(eval_folder, 'evaluation_data.npz'), 
             rewards=rewards, scores=scores, actions=actions, q_values=q_values, eval_time=eval_time)
    logging.info(f"Raw evaluation data saved in {eval_folder}")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_folder = os.path.join("evaluation_metrics", f"eval_{timestamp}")
    os.makedirs(eval_folder, exist_ok=True)

    log_file = os.path.join(eval_folder, 'evaluation.log')
    setup_logging(log_file)

    # Update vision model and trained agent path here
    model_path = "trained_agents/agent_18_08_2024_00_56 12Hour/final_model.zip"
    vision_model_path = "vision_models/FinalModel198Epochs.pt"
    
    # Loads the trained DQN agent
    model = DQN.load(model_path)
    env = IdleGameEnvironment(vision_model_path, "IdleGame")
    
    logging.info("Evaluating model...")
    logging.info("Press Ctrl+Shift+S at any time to force save and stop the evaluation.")
    rewards, scores, actions, q_values, eval_time = evaluate_model(model, env, total_steps=10000, log_interval=100)
    
    save_results(eval_folder, rewards, scores, actions, q_values, eval_time)

if __name__ == "__main__":
    main()