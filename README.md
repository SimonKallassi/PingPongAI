# Deep Q-Learning Pong AI

A PyTorch implementation of a Deep Q-Learning Network (DQN) that learns to play Pong from scratch. The AI learns through experience, developing strategies to compete against a rule-based opponent.
I decided a while back that I wanted to dive innto machine learning and have been watching some videos and reading some ducomentation about the topic. This is my first project using neural networks and deep learning.
Many more projects to come.
Hopefully I can create an AI for a 3D game someday.
You never know!
Enjoy and have fun using this!

## Demo


https://github.com/user-attachments/assets/e91bbe25-ccd6-4efc-977c-ace9e08d3356


## Features
- Real-time visualization of the game and training process
- Dynamic speed control (up to 10,000 FPS)
- Live statistics tracking (games played, rally length, etc.)
- Toggle-able display for faster training
- Automated save system for training checkpoints

## Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- CUDA-capable GPU (optional, for faster training)

### Development Environment
- Python IDE
- Git (for version control) - This really helps to be able to go back to earlier versions of your code if you mess up.
- Virtual Environment (recommended)

## Installation

1. Clone the repository
```bash
[https://github.com/SimonKallassi/PingPongAI.git]
cd PingPongAI
```

2. Create and activate a virtual environment (recommended)
```bash
# Windows
python -m venv env
.\env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the AI
```bash
# Start training with display
python main.py

# Start training without display (faster)
python main.py --no-render

# Continue training from a saved model
python main.py --load saved_models/model_[episode].pt
```

### Controls
- `0`: Toggle display on/off
- `UP`: Increase game speed
- `DOWN`: Decrease game speed
- `Ctrl+C`: Save and quit

## Project Structure
```
pong-dqn-ai/
├── main.py           # Main training script
├── pong_env.py       # Pong environment implementation
├── dqn_agent.py      # DQN agent implementation
├── requirements.txt  # Project dependencies
├── saved_models/    # Saved model checkpoints
└── README.md        # Project documentation
```

## Training Process

The AI uses the following key components:
- Deep Q-Network (DQN) with experience replay
- Epsilon-greedy exploration strategy
- Target network for stable learning
- Advanced reward shaping for better learning signals

Training parameters:
- Learning rate: 0.0005
- Epsilon decay: 0.99998
- Gamma (discount factor): 0.99
- Replay buffer size: 500,000 experiences
- Batch size: 256
- Please feel free to play around with these values to get the outcome you are looking for. These values are what I used to get a highscore of 60 relays in 800 games.
  
## Results
I played around with the parameters and was able to reach a 60 relay max score in around 800 games played.
If you can beat that open an issue I would love to see the new highscores with the changes you made!

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Acknowledgments
- OpenAI Gym/Gymnasium for the environment structure
- PyTorch for the deep learning framework

## Author
Simon El Kallassi

## Contact
- GitHub: [SimonKallassi](https://github.com/SimonKallassi)
- Email: kallassisimon@gmail.com
