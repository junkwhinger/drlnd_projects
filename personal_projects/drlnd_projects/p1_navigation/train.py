import os
import argparse
from collections import deque
import pickle

import numpy as np
import torch
from unityagents import UnityEnvironment

from model import net, buffer, agent
import utils


def run_training(agent, params):

    output_path = os.path.join("experiments", params.experiment_dir)

    scores = []
    scores_window = deque(maxlen=100)
    eps = params.eps_start
    initial_threshold = params.save_score_threshold

    env = UnityEnvironment(file_name=params.appname)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    for i_episode in range(1, params.n_episodes):

        completion = i_episode / params.n_episodes

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0

        while True:

            action = agent.act(state, eps)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done, completion)

            state = next_state

            score += reward

            if done:
                break

        scores_window.append(score)


        scores.append(score)

        eps = max(params.eps_end, params.eps_decay * eps)

        # display metrics
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        # save model if the latest average score is higher than 200.0
        if np.mean(scores_window) >= initial_threshold:
            print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                          np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '{}/cp{:04d}.pth'.format(output_path, i_episode - 100))
            initial_threshold += params.threshold_interval

    with open(os.path.join(output_path, "score.pkl"), "wb") as f:
        pickle.dump(scores, f)




def main(params):

    #TODO: make a network
    local_network = net.dqn_network(params).to(params.device)
    target_network = net.dqn_network(params).to(params.device)

    #TODO: make a replay buffer
    if params.use_per:
        replayBuffer = buffer.PrioritizedReplayBuffer(params)
    else:
        replayBuffer = buffer.ReplayBuffer(params)


    #TODO: define agent
    if params.use_ddqn:
        train_agent = agent.DDQNAgent(params, local_network, target_network, replayBuffer)
    else:
        train_agent = agent.DQNAgent(params, local_network, target_network, replayBuffer)

    #TODO: run training
    run_training(train_agent, params)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run DQN on Banana.app")
    parser.add_argument('--experiment-dir', type=str, default='ddqn_dn_netA_per')
    args = parser.parse_args()

    params = utils.Params(args.experiment_dir)
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.experiment_dir = args.experiment_dir

    main(params)