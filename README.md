## connect_four
### Features
#### Network
- Similar to the one used in AlphaZero but with 5 (instead of 19) residual blocks with 64 (instead of 256) filters
- GELU instead of ReLU activation ([Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415v5))
#### Training
- Gumbel AlphaZero Monte-Carlo Tree Search (MCTS) used for self-play and evaluation games
  ([Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO))
  - Compared to the original AlphaZero search algorithm, the Gumbel version ensures a policy improvement, even with
    as few as 2 simulations 
  - As such, only 32 (instead of 800) simulations were used per action during self-play
- _Late-To-Early Simulation Focus (LATE)_ enhancement
  ([Expediting Self-Play Learning in AlphaZero-Style Game-Playing Agents](https://www.researchgate.net/publication/374297603_Expediting_Self-Play_Learning_in_AlphaZero-Style_Game-Playing_Agents))
  - Losses weighted by game phase and training stage to prioritize learning late-game positions at the beginning of training (which was found to speed up training)
- 1cycle learning rate schedule instead of the original stepped learning rate schedule
  ([A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820))
  - Allowed a similar strength network to be trained in half the number of training steps
    - In the end, only 25,000 (instead of 700,000) training steps were needed
  - Larger learning rate made L2 regularization unnecessary; it was thus removed from the training process
#### Testing
- Round-robin tournament between runs with relative ELO table
- Relative ELO between consecutive test steps
- Interactive Jupyter notebook where one can play against the engine (with auto-generated animation of played game)
#### Game-specific
- Data augmentation through pseudo-random horizontal flipping of game observation and policy
- Armageddon mode
  - Since Connect Four is a solved game and the first player can always force a win, during training, a draw is treated as a loss for the first player and a win for the second
  - **+88 ELO** (compared to run1_baseline)
#### Miscellaneous
- Tensorboard visualizations for each run as training progresses
  - Each layer's weights
  - Training losses
  - Self-play ELO
  - Raw (without MCTS) network probabilities of selecting each action from the starting position
  - Raw network value estimate of the starting position (1 = win, 0 = draw, -1 = loss)
- Automatic saving/loading of trained weights
- Automatic periodic saving of replay buffer to enable pausing/continuing training
- Logging to console and log file

### Organization
#### Directories
- `animations`
  - SVG animations of sample self-play games at different checkpoints by training run
- `checkpoints`
  - Checkpoints by training run
- `networks`
  - `config.py`: Training run configuration
  - `network.py`: Network configuration
  - `network.txt`: Output of `flax.linen.tabulate` including a breakdown of the network's FLOPs by layer
  - `<run_name>.log`: Contains raw logs from the training run
- `tensorboard`
  - Tensorboard data files by training run
#### Files
- Training scripts
  - `calculate_flops.py`: Logs FLOPs table from `flax.linen.tabulate`
  - `common.py`: Contains global variables and common classes
  - `config.py`: Automatically copied from the latest network's folder when running `3_run_trainer.sh`
  - `evaluate.py`: Loads two networks (with potentially different architecture and configuration), pits them
    against each other, and outputs their ELO difference
  - `main.py`: Main training code
  - `network.py`: Automatically copied from the latest network's folder when running `3_run_trainer.sh`
  - `play.ipynb`: Interactive Jupyter notebook where one can play against the engine
    - Demo game is that of the `run2_armageddon` network (as the first player) against a
      [perfect solver](https://connect4.gamesolver.org/) (as the second player)
- Utility scripts
  - `1_disable_tmux.sh`: Disable tmux on remote systems
  - `2_install_requirements.sh`: Automatically installs requirements for training on remote systems
  - `3_run_trainer.sh`: Helper script for starting the training
  - `bayeselo`: Statically-linked BayesElo executable used by other scripts for calculating ELO
  - `calc_bayeselo.sh`: Prints BayesElo-calculated ratings table
  - `calc_elo_delta.sh`: Parses BayesElo output and prints ELO change and uncertainties
  - `run_tournament.sh`: Runs a round-robin tournament between networks
  - `upload_run.sh`: Helper script for automatically adding a new run's files to a new Git commit
- Resource files
  - `connect_four.svg`: SVG file of the demo game displayed in the Jupyter notebook
- Miscellaneous
  - `.gitignore`: Skips `__pycache__` and `.venv` folders, and all `.pyc` and `.pkl` files
  - `requirements.txt`: Python requirements file
  - `<latest_run_name>.log`: Output of network's MCTS-assisted action probabilities and value estimate from
    the demo Jupyter notebook

### Training environment
Tested on Ubuntu 22.04 with Python 3.10 and CUDA 12.2.

### To-dos
- Remove the unnecessary bias terms from all layers that precede a `BatchNorm` layer (`use_bias=False`)
- Add a `--round-robin` flag to `evaluate.py` and remove `run_tournament.sh` (prevent
  JIT recompilation every match)
- Optimize performance bottlenecks in the training pipeline

### License

Copyright © 2024 Bill Chow. All rights reserved.

This repository is made available for evaluation purposes by authorized individuals only. 
Unauthorized use, reproduction, modification, or distribution of this code is strictly prohibited.
