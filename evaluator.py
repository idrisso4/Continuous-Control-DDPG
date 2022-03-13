import time
from collections import deque

import numpy as np
import torch


def evaluate(
    env,
    brain_name,
    actor,
    device,
    n_episodes=100,
    max_t=1000,
):
    """Evaluation method.
    Params
    ======
        env: Unity Environment
        brain_name (str): name of the brain
        actor: the trained actor
        device: device cpu or gpu
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []
    scores_window = deque(maxlen=100)
    for _ in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            actor.eval()
            with torch.no_grad():
                action = actor(state).cpu().data.numpy()
            actor.train()
            action = np.clip(action, -1, 1)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            state = next_state
            score += reward
            if done:
                break
            time.sleep(0.05)
        scores_window.append(score)
        scores.append(score)
        print(
            "\nAfter 100 episodes!\tAverage Score: {:.2f}".format(
                np.mean(scores_window)
            )
        )

    return scores
