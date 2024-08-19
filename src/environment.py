import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyautogui
from collections import defaultdict
from src.detector import VisionCapture

class IdleGameEnvironment(gym.Env):
    def __init__(self, vision_model_path, game_name='IdleGame'):
        super().__init__()
        self.vision_capture = VisionCapture(vision_model_path)
        self.game_name = game_name
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Assuming 4 possible actions (clickable buttons)
        self.observation_space = spaces.Box(low=0, high=1920, shape=(5,), dtype=np.float32)
        
        self.last_score = 0
        self.clickable_classes = ['primary_btn', 'secondary_btn', 'building_btn', 'upgrade_btn']
        self.action_history = defaultdict(int)  # To track the number of times each action is taken
        self.step_count = 0
        self.total_reward = 0

    def step(self, action):
        self.step_count += 1
        # Ensure action is an integer
        action = int(action)
        
        print(f"Step {self.step_count}: Taking action {action}")
        
        # Take the specified action
        self.take_action(action)
        
        # Get the new observation
        observation = self.get_observation()
        
        # Calculate the reward based on the observation
        reward = self.calculate_reward(observation, action)
        self.total_reward += reward
        
        # Assume the episode is never done in this example; modify if needed
        done = False
        
        # Collect additional info, e.g., the current score
        info = {'score': observation[-1]}
        
        print(f"Step {self.step_count}: Reward = {reward}, Total Reward = {self.total_reward}, Score = {info['score']}, Observation = {observation}")
        
        # Return the observation, reward, done flag, truncated flag, and info
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        print("Resetting environment")
        
        # Get the initial observation
        observation = self.get_observation()
        
        # Initialise the last score with the initial score
        self.last_score = observation[-1]
        self.action_history.clear()  # Clear the action history on reset
        self.step_count = 0
        self.total_reward = 0
        
        # Collect initial info
        info = {'score': self.last_score}
        
        print(f"Reset: Initial observation = {observation}, Initial score = {info['score']}")
        
        return observation, info

    def take_action(self, action):
        # Capture the screen and detect objects
        _, detections = self.vision_capture.get_detected_frame_and_score_width()
    
        # Ensure detections is a list or another iterable
        if not isinstance(detections, list):
            detections = []  # Fallback to an empty list if detections is not as expected
    
        # Filter detections for clickable objects
        clickable_detections = [
            tuple(d) for d in detections if self.vision_capture.model.names[int(d[5])] in self.clickable_classes
        ]
    
        print(f"Clickable detections: {clickable_detections}")
    
        # Perform the action by clicking on the appropriate detection
        if action < len(clickable_detections):
            x1, y1, x2, y2, _, _ = clickable_detections[action]
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            print(f"Clicking at ({center_x}, {center_y})")
            pyautogui.click(center_x, center_y)
        else:
            print(f"Action {action} is out of range. No click performed.")

    def get_observation(self):
        # Capture the screen and detect objects
        _, detections = self.vision_capture.get_detected_frame_and_score_width()
    
        # Ensure detections is a list or iterable structure
        if not isinstance(detections, list):
            detections = []  # Fallback to an empty list if detections is not as expected
    
        # Convert the detections to a state
        state = self.detections_to_state(detections)
        print(f"Current state: {state}")
        return state

    def detections_to_state(self, detections):
        state = np.zeros(5, dtype=np.float32)
        score = 0

        # Process each detection
        for det in detections:
            if isinstance(det, np.ndarray):
                det = tuple(det.tolist())  # Convert to a tuple after converting to a list
            
            if not isinstance(det, (tuple, list)) or len(det) != 6:
                continue  # Skip if detection is not in expected format

            x1, y1, x2, y2, conf, cls = det
            class_name = self.vision_capture.model.names[int(cls)]
        
            # Set the state for clickable objects
            if class_name in self.clickable_classes:
                state[self.clickable_classes.index(class_name)] = 1
        
            # Use the width directly as the score
            elif class_name == 'score':
                score = x2 - x1

        # Use the score directly without normalisation
        state[-1] = score

        return state

    def calculate_reward(self, observation, action):
        current_score_width = observation[-1]
        width_increase = current_score_width - self.last_score

        # Apply a scaling factor to the width increase as the pixel width would be very small
        scaling_factor = 0.5
        scaled_width_increase = width_increase * scaling_factor

        # Ensure no negative rewards
        reward = max(scaled_width_increase, 0)

        # Cumulative Reward for Resource Management
        # Adjust these values to encourage more diverse actions
        if action == self.clickable_classes.index('upgrade_btn'):
            reward += 2.0  # Increased from 0.8
        if action == self.clickable_classes.index('building_btn'):
            reward += 1.5  # Increased from 0.4
        if action == self.clickable_classes.index('secondary_btn'):
            reward += 1.2  # Increased from 0.9
        if action == self.clickable_classes.index('primary_btn'):
            reward += 0.05  # Decreased from 0.1

        # Exploration Reward with increased impact
        exploration_bonus = 2 / (self.action_history[action] + 1)  # Increased from 1
        reward += exploration_bonus

        # New: Diversity bonus
        unique_actions = len(set(list(self.action_history.values())[-5:]))  # Last 5 actions
        diversity_bonus = unique_actions * 0.3
        reward += diversity_bonus

        # Update last score width
        self.last_score = current_score_width

        # Track the action in history
        self.action_history[action] += 1

        print(f"Reward calculation: width_increase = {width_increase}, scaled_increase = {scaled_width_increase}, "
            f"action_bonus = {reward - scaled_width_increase - exploration_bonus - diversity_bonus:.2f}, "
            f"exploration_bonus = {exploration_bonus:.2f}, diversity_bonus = {diversity_bonus:.2f}, "
            f"final_reward = {reward:.2f}")

        return reward