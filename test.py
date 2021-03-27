import torch
import numpy as np
from time import time
import os
from collections import namedtuple
import time

from utils import ReplayBuffer, MLPActorCritic
from gac import GAC

from osim.env import L2M2019Env


def load_pytorch_policy(fpath, device, deterministic=True):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = os.path.join(fpath)
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname, map_location=torch.device(device))

    print("obs_mean=" + str(model.obs_mean))
    print("obs_std =" + str(model.obs_std))

    # make function for producing an action given a single state
    def get_action(o):
        with torch.no_grad():
            o = torch.FloatTensor(o.reshape(1, -1)).to(device)
            action = model.act(o, deterministic)
        return action

    return get_action

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    o, r, d, ep_ret, ep_len, n = env.reset(obs_as_dict=False), 0, False, 0, 0, 0
    o = np.array(o)

    while n < num_episodes:
        # if render:
        #     env.render()
        #     time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a, obs_as_dict=False)
        o = np.array(o)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(obs_as_dict=False), 0, False, 0, 0
            o = np.array(o)
            n += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--difficulty', type=int, default=1)
    args = parser.parse_args()
    env = L2M2019Env(visualize=args.render, difficulty=args.difficulty)
    get_action = load_pytorch_policy(args.fpath, args.device, args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, args.render)