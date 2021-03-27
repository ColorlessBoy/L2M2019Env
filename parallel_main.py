import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.rpc import rpc_sync

import gym
import numpy as np

import torch
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from torch.distributions import Categorical

import argparse
import torch.distributed.rpc as rpc

import os
from itertools import count

import torch.multiprocessing as mp

import csv
import json
from collections import namedtuple

from utils import ReplayBuffer, MLPActorCritic
from gac import GAC
from osim.env import L2M2019Env

from time import time

AGENT_NAME = "agent"
OBSERVER_NAME = "obs{}"

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

class Observer:

    def __init__(self, args):
        self.id = rpc.get_worker_info().id

        if args.env_name == 'L2M2019Env':
            self.env = L2M2019Env(visualize=False, difficulty=args.difficulty, seed=args.seed+self.id)
            self.test_env = L2M2019Env(visualize=False, difficulty=args.difficulty, seed=args.seed+self.id+999)
        else:
            self.env = gym.make(args.env_name)
            self.test_env = gym.make(args.env_name)
            self.env.seed(args.seed+self.id)
            self.test_env.seed(args.seed+self.id+999)
        
        if args.env_name == 'L2M2019Env':
            self.obs = np.array(self.env.reset(obs_as_dict=False))
        else:
            self.obs = self.env.reset()

        self.act_limit = self.env.action_space.high[0]
        self.done = False
        self.len = 0

        self.args = args

        print("observer")

    def run_episode(self, agent_rref, n_steps, random):
        for step in range(n_steps):
            if self.done:
                if self.args.env_name == 'L2M2019Env':
                    self.obs = np.array(self.env.reset(obs_as_dict=False))
                else:
                    self.obs = self.env.reset()
                self.len, self.done = 0, False

            # send the state to the agent to get an action
            if random:
                a = self.env.action_space.sample()
                if self.args.env_name != 'L2M2019Env':
                    a /= self.act_limit
            else:
                a = _remote_method(Agent.select_action, agent_rref, self.obs)

            # apply the action to the environment, and get the reward
            # [-1, 1] => [0, 1]
            if self.args.env_name == 'L2M2019Env':
                o2, r, self.done, _ = self.env.step(np.abs(a), obs_as_dict=False)
                o2 = np.array(o2)
            else:
                o2, r, self.done, _ = self.env.step(a * self.act_limit)

            self.len += 1

            # report the reward to the agent for training purpose
            if self.done == True and self.len == self.args.max_ep_len:
                _remote_method(Agent.add_memory, agent_rref, self.id, self.obs, a, r, o2, False)
            else:
                _remote_method(Agent.add_memory, agent_rref, self.id, self.obs, a, r, o2, self.done)

            self.obs = o2
    
    def test_episode(self, agent_rref):
        if self.args.env_name == 'L2M2019Env':
            o = np.array(self.test_env.reset(obs_as_dict=False))
        else:
            o = self.test_env.reset()

        ep_ret, ep_len, d = 0.0, 0.0, False
        while not d and ep_len < self.args.max_ep_len:
            a = _remote_method(Agent.select_action, agent_rref, o, True)
            if self.args.env_name == 'L2M2019Env':
                o, r, d, _ = self.test_env.step(np.abs(a), abs_as_dict=False)
                o = np.array(o)
            else:
                o, r, d, _ = self.test_env.step(a)
            ep_ret += r
            ep_len += 1

        _remote_method(Agent.add_test_data, agent_rref, ep_ret, ep_len)
        
class Agent:
    def __init__(self, world_size, args):
        if args.env_name == 'L2M2019Env':
            env = L2M2019Env(visualize=False, difficulty=args.difficulty)
        else:
            env = gym.make(args.env_name)

        obs_dim  = env.observation_space.shape[0]
        act_dim  = env.action_space.shape[0]

        self.device = torch.device(args.device)

        self.args = args
        self.world_size = world_size

        self.actor_critic = MLPActorCritic(obs_dim, act_dim, hidden_sizes=args.hidden_sizes).to(self.device)
        self.replay_buffer = [ReplayBuffer(obs_dim, act_dim, args.buffer_size) for _ in range(1, world_size)]

        if args.env_name == 'L2M2019Env':
            with open('obs.npy', 'rb') as f:
                obs_mean = np.load(f)
                obs_std  = np.load(f)
            
            for rb in range(self.replay_buffer):
                rb.obs_mean = obs_mean
                rb.obs_std  = obs_std

            self.actor_critic.obs_mean = torch.from_numpy(obs_mean)
            self.actor_critic.obs_std  = torch.from_numpy(obs_std)

        self.gac = GAC(self.actor_critic, self.replay_buffer, device=self.device, gamma=args.gamma,
              alpha_start=args.alpha_start, alpha_min=args.alpha_min, alpha_max=args.alpha_max)

        self.test_len = 0.0
        self.test_ret = 0.0

        self.ob_rrefs = []
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer, args=(args,)))

        self.agent_rref = RRef(self)

    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        a = self.actor_critic.act(obs, deterministic)
        return a

    def add_memory(self, ob_id, o, a, r, o2, d):
        self.replay_buffer[ob_id-1].store(o, a, r, o2, d)

    def run_episode(self, n_steps=0, random=False):
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps, random)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def add_test_data(self, ret, length):
        self.test_ret += ret
        self.test_len += length

    def test_episode(self):
        futs, self.test_ret, self.test_len= [], 0.0, 0.0
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.test_episode, ob_rref, self.agent_rref)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

        self.test_ret /= (self.world_size - 1)
        self.test_len /= (self.world_size - 1)
        return self.test_ret, self.test_len
    
    def update(self):
        for _ in range(self.args.steps_per_update):
            loss_a, loss_c, alpha = self.gac.update(self.args.batch_size)
        self.gac.update_beta()
        print("loss_actor = {:<22}, loss_critic = {:<22}, alpha = {:<20}, beta = {:<20}".format(loss_a, loss_c, alpha, self.gac.beta))

def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        logdir = "./data/gac/{}/{}-seed{}-{}".format(args.env_name, args.env_name, args.seed, time())
        config_name = 'config.json'
        file_name = 'progress.csv'
        model_name = 'model.pt'
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        config_json = json.dumps(args._asdict())
        config_json = json.loads(config_json)
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        with open(os.path.join(logdir, config_name), 'w') as out:
            out.write(output)

        full_name = os.path.join(logdir, file_name)
        csvfile = open(full_name, 'w')
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['TotalEnvInteracts', 'AverageTestEpRet', 'AverageTestEpLen'])

        full_model_name = os.path.join(logdir, model_name)

        agent = Agent(world_size, args)

        agent.run_episode(args.start_steps, True)

        for t1 in range(args.total_epoch):
            for t2 in range(int(args.steps_per_epoch / args.steps_per_update)):
                agent.run_episode(args.steps_per_update)
                agent.update()

            test_ret, test_len = agent.test_episode() 

            t = t1*args.steps_per_epoch + (t2 + 1)*args.steps_per_update

            print("Step {:>10}: test_ret = {:<20}, test_len = {:<20}".format(t, test_ret, test_len))
            print("-----------------------------------------------------------")

            writer.writerow([t, test_ret, test_len])
            csvfile.flush()
            torch.save(agent.actor_critic, full_model_name)
    else:
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)

    rpc.shutdown()

Args = namedtuple('Args',
               ('alg_name',
                'env_name', 
                'device', 
                'seed', 
                'hidden_sizes', 
                'buffer_size',
                'epoch_per_test',
                'max_ep_len', 
                'total_epoch', 
                'steps_per_epoch',
                'start_steps',
                'reward_scale',
                'update_after',
                'steps_per_update',
                'batch_size',
                'alpha_start',
                'alpha_min',
                'alpha_max',
                'difficulty',
                'gamma'))

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--env', default='L2M2019Env', metavar='G',
                        help='name of environment name (default: HalfCheetah-v3)')
    parser.add_argument('--device', default='cuda:0', metavar='G',
                        help='device (default cuda:0)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='total epochs(default: 1000)')
    parser.add_argument('--reward_scale', type=float, default=1.0, metavar='N',
                        help='reward_scale (default: 1.0)')
    parser.add_argument('--alpha_start', type=float, default=1.2, metavar='N',
                        help='alpha_start (default: 1.2)')
    parser.add_argument('--alpha_min', type=float, default=1.0, metavar='N',
                        help='alpha_min (default: 1.0)')
    parser.add_argument('--alpha_max', type=float, default=1.5, metavar='N',
                        help='alpha_max (default: 1.5)')
    parser.add_argument('--difficulty', type=int, default=1, metavar='N',
                        help='difficulty for L2M2019Env(default: 1)')
    parser.add_argument('--max_ep_len', type=int, default=1000, metavar='N',
                        help='max_ep_len(default: 1000)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='N',
                        help='gamma (default: 0.99)')
    parser.add_argument('--observer', type=int, default=10, metavar='N',
                        help='observer number(default: 10)')
    
    args = parser.parse_args()

    alg_args = Args("gac",          # alg_name
                args.env,           # env_name
                args.device,        # device
                args.seed,          # seed
                [400, 300],         # hidden_sizes
                int(1e6),           # replay buffer size
                10,                 # epoch per test
                args.max_ep_len,    # max_ep_len
                args.epochs,        # total epochs
                4000,               # steps per epoch
                1000,               # start steps
                args.reward_scale,  # reward scale 
                1000,               # update after
                50,                 # steps_per_update
                100,                # batch size
                args.alpha_start,
                args.alpha_min,
                args.alpha_max,
                args.difficulty,
                args.gamma)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # to call a function on an rref, we could do the following
    # _remote_method(some_func, rref, *args)
    mp.spawn(
        run_worker,
        args=(args.observer+1, alg_args),
        nprocs=args.observer+1,
        join=True
    )
