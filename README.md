# Continuous Control DDPG Project

================================================================================


This project is part of Udacity Deep Reinforcement Learning Nanodegree, which is a four-month course that I am enrolled in.
The purpose of this project is to train a DDPG agent for Unity ML-Agents Reacher environment.


## Environment

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher) environment.


![Environment](reacher.gif)

In this environment, a double-jointed arm can move to target locations.
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


## Install


### Clone the repo

```
git clone https://github.com/idrisso4/Continuous-Control-DDPG

cd Continuous-Control-DDPG
```

### Create conda Environment

```
# Create the conda environment
conda create -n deeprlnd python=3.6 numpy=1.13.3 scipy

# Activate the new environment
source activate deeprlnd

# Install dependencies
conda install pytorch torchvision -c pytorch
pip install matplotlib
pip install unityagents==0.4.0
```

### Download the Unity Environment

#### One Agent:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

#### Multi Agent:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Config

Change the parameters in the config.yaml file
(you must change this variable ENVIRONMENT to the location of your downloaded environment)

## Train the Agent:

```
python main.py --train
```

## Evaluate the agent:

```
python main.py --eval
```
