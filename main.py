import gym
import torch
import numpy as np
from time import time
import os
import csv
import json
from collections import namedtuple

from utils import ReplayBuffer, MLPActorCritic
from gac import GAC

from osim.env import L2M2019Env

def main(args):

    if 'L2M2019Env' in args.env_name:
        env = L2M2019Env(visualize=False, difficulty=args.difficulty)
        test_env = L2M2019Env(visualize=False, difficulty=args.difficulty)
    else:
        env = gym.make(args.env_name)
        test_env = gym.make(args.env_name)
    device = torch.device(args.device)

    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    test_env.seed(args.seed + 999)

    # 2.Create actor, critic, EnvSampler() and PPO.
    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    act_high = env.action_space.high
    act_low  = env.action_space.low
    obs_high = env.observation_space.high
    obs_low  = env.observation_space.low

    actor_critic = MLPActorCritic(obs_dim, act_dim, hidden_sizes=args.hidden_sizes).to(device)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)
    replay_buffer.obs_mean = (obs_high + obs_low) / 2
    replay_buffer.obs_std  = (obs_high - obs_low) / 2

    actor_critic.obs_mean = torch.FloatTensor(replay_buffer.obs_mean).to(device)
    actor_critic.obs_std  = torch.FloatTensor(replay_buffer.obs_std).to(device)

    gac = GAC(actor_critic, replay_buffer, device=device, gamma=args.gamma,
              alpha_start=args.alpha_start, alpha_min=args.alpha_min, alpha_max=args.alpha_max)
    

    def act_encoder(y):
        # y = [min, max] ==> x = [-1, 1]
        return (y - act_low) / (act_high - act_low) * 2.0 - 1.0
    
    def act_decoder(x):
        # x = [-1, 1] ==> y = [min, max]
        return (x + 1.0) / 2.0 * (act_high - act_low) - act_low

    def reward_shaping(env):

        reward = 20.0

        # Reward for not falling down
        state_desc = env.get_state_desc()
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = env.vtgt.get_vtgt(p_body).T

        vel_penalty = np.linalg.norm(v_body - v_tgt)

        muscle_penalty = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            muscle_penalty += np.square(
                state_desc['muscles'][muscle]['activation'])

        ret_r = reward - (vel_penalty * 3 + muscle_penalty * 1)

        if vel_penalty < 0.3:
            ret_r += 20

        return ret_r

    # 3.Start training.
    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = actor_critic.act(o, deterministic)
        return a

    def test_agent():
        test_ret, test_len = 0, 0
        for j in range(args.epoch_per_test):
            o, d, ep_ret, ep_len = test_env.reset(obs_as_dict=False), False, 0, 0
            o = np.array(o)
            while not(d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time 
                a = get_action(o, True)

                o, r, d, _ = test_env.step(act_decoder(a), obs_as_dict=False)
                o = np.array(o)
                ep_ret += r
                ep_len += 1

            test_ret += ep_ret
            test_len += ep_len
        return test_ret / args.epoch_per_test, test_len / args.epoch_per_test

    total_step = args.total_epoch * args.steps_per_epoch
    o, d, ep_ret, ep_len = env.reset(obs_as_dict=False), False, 0, 0
    o = np.array(o)
    for t in range(1, total_step+1):
        if t <= args.start_steps:
            a = act_encoder(env.action_space.sample())
        else:
            a = get_action(o, deterministic=False)
        
        o2, r, d, _ = env.step(act_decoder(a), obs_as_dict=False)
        o2 = np.array(o2)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)

        d = False if ep_len==args.max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, reward_shaping(env) * args.reward_scale, o2, d)

        o = o2
        if d or (ep_len == args.max_ep_len):
            o, ep_ret, ep_len = env.reset(obs_as_dict=False), 0, 0
            o = np.array(o)

        if t >= args.update_after and t % args.steps_per_update==0:
            gac.update_obs_param()
            for _ in range(args.steps_per_update):
                loss_a, loss_c, alpha = gac.update(args.batch_size)
            gac.update_beta()
            print("loss_actor = {:<22}, loss_critic = {:<22}, alpha = {:<20}, beta = {:<20}".format(loss_a, loss_c, alpha, gac.beta))

        # End of epoch handling
        if t >= args.update_after and t % args.steps_per_epoch == 0:
            test_ret, test_len = test_agent()
            print("Step {:>10}: test_ret = {:<20}, test_len = {:<20}".format(t, test_ret, test_len))
            print("-----------------------------------------------------------")
            yield t, test_ret, test_len, actor_critic

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
                'frame_skip',
                'gamma'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--env', default='HalfCheetah-v3', metavar='G',
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

    
    args = parser.parse_args()

    alg_args = Args("gac",          # alg_name
                args.env,           # env_name
                args.device,        # device
                args.seed,          # seed
                [800, 400, 200],    # hidden_sizes
                int(1e6),           # replay buffer size
                10,                 # epoch per test
                2500,               # max_ep_len
                args.max_ep_len,    # total epochs
                4000,               # steps per epoch
                10000,              # start steps
                args.reward_scale,  # reward scale 
                1000,               # update after
                50,                 # steps_per_update
                100,                # batch size
                args.alpha_start,
                args.alpha_min,
                args.alpha_max,
                args.difficulty,
                4,                  # frame_skip
                args.gamma)

    logdir = "./data/gac/{}/{}-seed{}-{}".format(alg_args.env_name, alg_args.env_name,alg_args.seed, time())
    config_name = 'config.json'
    file_name = 'progress.csv'
    model_name = 'model.pt'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    config_json = json.dumps(alg_args._asdict())
    config_json = json.loads(config_json)
    output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
    with open(os.path.join(logdir, config_name), 'w') as out:
        out.write(output)

    full_name = os.path.join(logdir, file_name)
    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['TotalEnvInteracts', 'AverageTestEpRet', 'AverageTestEpLen'])

    full_model_name = os.path.join(logdir, model_name)

    for t, reward, len, model in main(alg_args):
        writer.writerow([t, reward, len])
        csvfile.flush()
        torch.save(model, full_model_name)
    
    csvfile.close()