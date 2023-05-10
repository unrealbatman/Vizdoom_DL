<div align="center">

# <b>Value-Based Reinforcement Learning using DQN</b>

</div>

The project explores the application of Deep Q-networks (DQNs) for value-based reinforcement learning in the context of
ViZDoom game environment. The objective is to train an agent to learn to maximize its rewards by making sequential decisions in an environment. This project uses the ViZDoom game environment to demonstrate the effectiveness of DQN in learning policies for the game.


## Introduction

In reinforcement learning, an agent learns to make decisions by interacting with an environment. In value-based
reinforcement learning, the agent learns to estimate the value of actions in the current state to maximize its rewards.
The deep Q-network algorithm combines deep learning and reinforcement learning to approximate the Q-value function,
which is the expected cumulative reward for taking a specific action in a given state.

In this project, we compare the performance of two different architectures of deep Q-networks (DQNs) for value-based reinforcement learning: **DQN** and **Duel DQN**.

The DQN architecture uses a single neural network to approximate the Q-value function. It takes the current state of the environment as input and outputs the estimated Q-values for all possible actions in that state. The agent selects the action with the highest Q-value to maximize its rewards.

<div align="center"><image width="300" src="./images/dqn.png"></div>

The Duel DQN architecture is an extension of the DQN architecture that separates the estimation of the state value and action advantage functions. It uses two parallel neural networks: one estimates the state value function, which measures the value of being in a particular state regardless of the action taken, and the other estimates the action advantage function, which measures the advantage of taking a particular action in a particular state over other possible actions. The Q-value function is then computed as the sum of the state value and action advantage functions.
<div align="center"><image width="300" src="./images/ddqn.png"></div>

## Results

To compare the performance of the two architectures, we train agents using both DQN and Duel DQN on the ViZDoom game
environment. We monitor the learning process and performance of the agents using mean reward per episode on both training
and testing game episodes.

The results show that the Duel DQN architecture outperforms the DQN architecture in terms of learning speed and
final performance. The Duel DQN agent is able to achieve a higher score in the game environment and learns a more
optimal policy faster than the DQN agent. This is likely due to the separation of the state value and action advantage
functions, which helps to reduce overestimation of the Q-values and improve the stability of the learning process.

**TODO: Update**

## Usage

To run the code, follow these steps:

1. Clone the repository
    ```bash
    git clone git@github.com:utsavoza/fuzzy-enigma.git
    ```
2. Setup and activate the virtual environment
    ```bash
    python3 -m venv .
    source ./bin/activate
    ```

3. Install the required dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Train the DQN agent
    ```bash
    python train.py
    ```
5. See the trained DQN agent in action
    ```bash
    python test.py
    ```

## References

- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)

## License

    Copyright (c) 2023 Utsav Oza

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.