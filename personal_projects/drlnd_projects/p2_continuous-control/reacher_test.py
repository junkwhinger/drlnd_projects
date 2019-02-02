import argparse

import torch
import numpy as np

from agent import DDPGAgent

from unityagents import UnityEnvironment


def main():
    parser = argparse.ArgumentParser(description="Run Reacher with a trained agent.")
    parser.add_argument("--agent-version", type=str, default="solved", help="type in the agent version")

    args = parser.parse_args()

    #load env
    env = UnityEnvironment(file_name='Reacher_multi.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load agent
    base_agent = DDPGAgent(random_seed=42,
                           device=device,
                           n_agents=num_agents,
                           state_size=state_size,
                           action_size=action_size,
                           buffer_size=int(1e5),
                           batch_size=128,
                           gamma=0.99,
                           tau=0.001,
                           lr_actor=1e-4,
                           lr_critic=1e-4,
                           weight_decay=0.0)

    #load params
    base_agent.actor_local.load_state_dict(torch.load("ckpt/actor_{}.pth".format(args.agent_version)))
    base_agent.critic_local.load_state_dict(torch.load("ckpt/critic_{}.pth".format(args.agent_version)))

    #run_test
    states = env_info.vector_observations
    scores = np.zeros(num_agents)

    while True:
        actions = base_agent.act(states)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    print("[{}] Total Score: {}".format(args.agent_version, np.mean(scores)))

if __name__ == "__main__":
    main()