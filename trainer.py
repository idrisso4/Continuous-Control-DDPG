from collections import deque

import numpy as np
import torch


def train(env, brain_name, agent, n_episodes=2000, max_t=1000):
    """Train Method.
    Params
    ======
        env: Unity Environment
        brain_name (str): name of the brain
        agent: object of the class Agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[
                brain_name
            ]  # send all actions to tne environment
            next_state = env_info.vector_observations[
                0
            ]  # get next state (for each agent)
            reward = env_info.rewards[0]  # get reward (for each agent)
            done = env_info.local_done[0]  # see if episode finished
            state = next_state
            agent.step(state, action, reward, next_state, done)
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print(
            "\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}".format(
                i_episode, np.mean(scores_deque), score
            ),
            end="",
        )
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
    return scores
