[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Discrete Banana Man
[Unity ML-Agents](https://unity3d.com/machine-learning) + [Deep Q-Network (DQN)](https://deepmind.com/research/dqn/) using [PyTorch](https://pytorch.org/).

Project on use of value-based methods in reinforcement learning for partial fulfillment of [Udacity's Deep Reinforcement Learning Nanodegree.](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

### Background

See attached ```report.pdf``` for background information on reinforcement learning and the Deep Q-Network architecture utilized in this repository.

### Introduction

The goal of the agent is to navigate through a cluttered, enclosed environment filled with nutritious yellow bananas, and poisonous blue bananas.

![Trained Agent][image1]

The agent's (i.e., Banana Man) goal is to maximize the intake of yellow bananas, while avoiding the blue bananas. For each consumed yellow banana, Banana Man will receive +1 point; for each blue banana, the agent will receive -1 points.

During an episode of the game, can we train our Banana Man to achieve a net score of +13 points averaged over 100 episodes? (the score necessary to consider the learning problem solved!)

In this modified version of [ML-Agent's Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) provided by Udacity, Banana Man can detect 37 features of its environment, including Banana Man's own velocity and the objects in his field of view (in his forward direction).

Banana Man can navigate through his environment by taking one of four discrete actions:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Getting Started
1. [Download the Anaconda Python Distribution](https://www.anaconda.com/download/)

2. Once installed, open the Anaconda command prompt window and create a new conda environment that will encapsulate all the required dependencies. Replace **"desired_name_here"** with whatever name you'd like. Python 3.6 is required.

    `conda create --name "desired_name_here" python=3.6`  
    `activate "desired_name_here"`

3. Clone this repository on your local machine.

    `git clone https://github.com/ShortFox/Discrete-Banana-Man.git`  

4. Navigate to the `python/` subdirectory in this repository and install all required dependencies

    `cd Discrete-Banana-Man/python`  
    `pip install .`  

    **Windows Users:** When installing the required dependencies, you may receive an error when trying to install "torch." If that is the case, then install pytorch manually, and then run `pip install .` again. Note that pytorch version 0.4.0 is required:

    `conda install pytorch=0.4.0 -c pytorch`

5. Download the Banana Man Unity environment from one of the links below.  Note you do not need to have Unity installed for this repository to function.

    Download the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

6. Place the unzipped file inside the repository location. Note that the Windows (x86_64) version of the environment is already included (must be unzipped).

### Instructions to train the Banana Man agent

1. Open `main.py` and edit the `file_name` variable to correctly locate the location of the Banana Man Unity environment.

2. Within the same anaconda prompt, return to the `Discrete-Banana-Man` subfolder, and then train the agent:

    `cd ..`  
    `python main.py`

3. Following training, `checkpoint.pth` will be created which represents the neural network's weights. Additionally, `banana_scores.csv` will contain the scores from training (with the row indicating episode number).

4. To watch your trained agent, first open `main.py` and edit the following line, changing the variable `train_agent`:

    `agent = Agent(environment = env, agent_name = brain_name, train_agent = False)``

    By default, `checkpoint.pth` will be used for the neural network's weights.

    Now, to watch your agent in real-time, execute `main.py`:

    `python main.py`

### Interested to make your own changes?

- `model.py` - defines the neural network architecture and contains the `save` and `load` functions to save/load the network's weights.
- `dqn_agent.py` - defines the agent training program. Contains the classes `Agent()`, which defines the agent's behavior and neural network update rules, and `ReplayBuffer()`, which keeps track of the agent's state-action-reward experiences in memory for randomized batch learning for the neural network.
- `main.py` - simple script defining the Unity environment and command to run Banana Man through his banana-filled world.

See comments in associated files for more details regarding specific functions/variables.
