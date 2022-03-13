from collections import deque

import numpy as np
import torch

from agent import Agent
from utils import read_config


def train(env, brain_name: str, agent_config: dict, n_episodes=2000, max_t=1000):
    """Train Method.
    Params
    ======
        env: Unity Environment
        brain_name (str): name of the brain
        agent_config (dict): config of the agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """

    config = read_config()

    env_info = env.reset(train_mode=True)[brain_name]

    avg_score = []
    scores_deque = deque(maxlen=100)
    scores = np.zeros(len(env_info.agents))

    env_info = env.reset(train_mode=True)[brain_name]

    states = env_info.vector_observations

    agents = [Agent(**agent_config) for _ in range(len(env_info.agents))]
    action = [agent.act(states[i]) for i, agent in enumerate(agents)]

    for i_episode in range(1, n_episodes + 1):
        states = env_info.vector_observations
        for agent in agents:
            agent.reset()
        for t in range(max_t):
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            step_t = zip(agents, states, actions, rewards, next_states, dones)

            for agent, state, action, reward, next_step, done in step_t:
                agent.memory.add(state, action, reward, next_step, done)
                if t % config["TIME_STEPS"] == 0:
                    agent.step(state, action, reward, next_step, done, config["UPDATE"])
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        score = np.mean(scores)
        avg_score.append(score)
        scores_deque.append(score)
        avg = np.mean(scores_deque)

        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode,
                avg,
            ),
            end="\n",
        )

        if np.mean(scores_deque) > 30.0:
            print(
                f"Enviroment solved in episode={i_episode} avg_score={avg:.2f}".format(
                    i_episode=i_episode, avg=avg
                )
            )

            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")

            return avg_score

    return avg_score
