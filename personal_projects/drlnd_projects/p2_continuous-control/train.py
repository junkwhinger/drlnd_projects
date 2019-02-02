import argparse
from collections import deque

import torch
import numpy as np

from agent import DDPGAgent

from unityagents import UnityEnvironment


def ddpg_train(env, brain_name, agent, n_agents, n_episodes, len_scores=100):
    # base_score
    base_score = 0

    # score logging
    scores = []
    scores_window = deque(maxlen=len_scores)

    # for every episode
    for episode in range(n_episodes):

        # reset env
        env_info = env.reset(train_mode=True)[brain_name]

        # get states
        states = env_info.vector_observations

        # reset agent
        agent.reset()

        # reset score
        score = np.zeros(n_agents)

        # run until game ends
        while True:

            # decide actions for the current state
            actions = agent.act(states)

            # execute actions
            env_info = env.step(actions)[brain_name]

            # get next_states, rewards, dones
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # learn the agent
            agent.step(states, actions, rewards, next_states, dones)

            # update score
            score += rewards

            # update current states with the next_states
            states = next_states

            # if any of the agents are done, break
            if np.any(dones):
                break

        scores.append(np.mean(score))
        scores_window.append(np.mean(score))

        # save actor and critic checkpoints
        if np.mean(scores_window) > base_score:
            agent.save_checkpoints("score{}".format(int(np.mean(scores_window))))
            base_score += 10

        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score),
                                                                                   np.mean(scores_window)), end="")

        if np.mean(scores_window) >= 30.0:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            agent.save_checkpoints("solved")
            break

    return agent, scores


def main():
    parser = argparse.ArgumentParser(description="Run Reacher with a trained agent.")
    parser.add_argument("--n-episodes", type=int, default=1000, help="number of episodes to run")
    parser.add_argument("--buffer-size", type=int, default=10000, help="size of replay buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ckpt", type=str, default="ckpt")

    args = parser.parse_args()

    # load env
    env = UnityEnvironment(file_name='Reacher_multi.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # params
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize agent
    agent = DDPGAgent(random_seed=42,
                      device=device,
                      n_agents=num_agents,
                      state_size=state_size,
                      action_size=action_size,
                      buffer_size=args.buffer_size,
                      batch_size=args.batch_size,
                      gamma=args.gamma,
                      tau=args.tau,
                      lr_actor=args.lr_actor,
                      lr_critic=args.lr_critic,
                      weight_decay=args.weight_decay,
                      checkpoint_folder=args.ckpt)

    # run training
    ddpg_train(env, brain_name, agent, n_agents=num_agents, n_episodes=args.n_episodes, len_scores=100)


if __name__ == "__main__":
    main()