import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from actor import Actor
from evaluator import evaluate
from trainer import train
from utils import read_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    config = read_config()

    device = torch.device(config["device"])

    env = UnityEnvironment(file_name=config["ENVIRONMENT"])

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)

    # examine the state space
    states = env_info.vector_observations
    print("States look like:", states)
    state_size = states.shape[1]
    print("States have length:", state_size)

    agent_config = {
        "state_size": state_size,
        "action_size": action_size,
        "random_seed": 0,
    }

    model_train = args.train
    model_eval = args.eval

    if model_train:

        scores = train(
            env=env,
            brain_name=brain_name,
            agent_config=agent_config,
            n_episodes=config["n_episodes"],
            max_t=config["max_t"],
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.show()

    if model_eval:

        model = Actor(state_size, action_size, 0).to(device)
        model.load_state_dict(torch.load("checkpoint_actor.pth"))
        model.eval()

        evaluate(env, brain_name, model, device, n_episodes=100, max_t=1000)
