# Idle-Game-Automation-Using-Computer-Vision-and-AI

Investigating the general automation of Cookie Clicker, Kiwi Clicker and Universal Paperclips using a single model.

This project uses reinforcement learning to automate gameplay in idle games.

## Setup

1. Install dependencies:
pip install -r requirements.txt

2. Ensure you have the trained object detection model in the `vision_models/` directory and update the model path.

## Usage

1. Train the AI:
python train.py

2. Play the game using the trained AI:
python evaluateAgent.py

3. New YOLOv8 models can be evaluated using the standalone script:
python evaluateVisionModel.py

## Project Structure

- `src/detector.py`: Script that captures screen and detects 
- `src/environment.py`: Gymnasium environment for the idle games.
- `train.py`: Script to train the RL agent.
- `evaluateAgent.py`: Script to use the trained agent to play the game.
- `evaluateVisionModel.py`: Script used to solely run the object detection model for evaluation.
- `trained_agents` : Contains the trained agents.
- `vision_models` : Contains the YOLOv8 Models.
- `evaluation_metrics` : Contains the results generated after letting a trained agent play.

## Notes

- Ensure the game window is visible and active when running the scripts.
- While testing the vision model, the window generated could be infinite. To avoid this, a multi-monitor setup is recommended.
- The AI uses screen capture and mouse clicks to interact with the game, so avoid using the computer for other tasks during training or gameplay.
- The project is trained on Cookie Clicker and has been tested on Universal Paper Clips and Kiwi Clicker.
