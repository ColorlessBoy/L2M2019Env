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

    data = np.load('./official_obs_scaler.npz')
    obs_mean, obs_std = data['mean'], data['std']

    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    test_env.seed(args.seed + 999)

    # 2.Create actor, critic, EnvSampler() and PPO.
    if 'L2M2019Env' in args.env_name:
        obs_dim = 99
    else:
        obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]

    act_high = env.action_space.high
    act_low  = env.action_space.low

    actor_critic = MLPActorCritic(obs_dim, act_dim, hidden_sizes=args.hidden_sizes).to(device)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)

    gac = GAC(actor_critic, replay_buffer, device=device, gamma=args.gamma,
              alpha_start=args.alpha_start, alpha_min=args.alpha_min, alpha_max=args.alpha_max)

    def act_encoder(y):
        # y = [min, max] ==> x = [-1, 1]
        # if args.env_name == 'L2M2019Env':
        #     return y
        return (y - act_low) / (act_high - act_low) * 2.0 - 1.0
    
    def act_decoder(x):
        # x = [-1, 1] ==> y = [min, max]
        # if args.env_name == 'L2M2019Env':
        #     return np.abs(x)
        return (x + 1.0) / 2.0 * (act_high - act_low) - act_low

    def get_observation(env):
        obs = np.array(env.get_observation()[242:])

        obs = (obs - obs_mean) / obs_std

        state_desc = env.get_state_desc()
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = env.vtgt.get_vtgt(p_body).T

        return np.append(obs, v_tgt)

    def get_reward(env):
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
    
    def generate_success(o, o2):
        reward = 40.0

        # Reward for not falling down
        state_desc = env.get_state_desc()
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]

        theta = np.random.random() * 2 * np.pi - np.pi
        radius = np.random.random() * 0.3
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius

        v_tgt = [v_body[0] + x, v_body[1] + y]

        vel_penalty = radius

        muscle_penalty = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            muscle_penalty += np.square(
                state_desc['muscles'][muscle]['activation'])

        ret_r = reward - (vel_penalty * 3 + muscle_penalty * 1)

        new_o = np.append(o[:-2], v_tgt)
        new_o2 = np.append(o2[:-2], v_tgt)

        return new_o, ret_r, new_o2

    # 3.Start training.
    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = actor_critic.act(o, deterministic)
        return a

    def test_agent():
        test_ret, test_len = 0, 0
        for j in range(args.epoch_per_test):
            _, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            o = get_observation(test_env)
            while not(d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time 
                a = get_action(o, True)

                _, r, d, _ = test_env.step(act_decoder(a))
                o = get_observation(test_env)
                ep_ret += r
                ep_len += 1

            test_ret += ep_ret
            test_len += ep_len
        return test_ret / args.epoch_per_test, test_len / args.epoch_per_test

    total_step = args.total_epoch * args.steps_per_epoch
    _, d, ep_ret, ep_len = env.reset(), False, 0, 0
    o = get_observation(env)
    for t in range(1, total_step+1):
        if t <= args.start_steps:
            a = act_encoder(env.action_space.sample())
        else:
            a = get_action(o, deterministic=False)
        
        _, r, d, _ = env.step(act_decoder(a))
        o2 = get_observation(env)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)

        d = False if ep_len==args.max_ep_len else d

        # if not d:
        #     new_o, new_r, new_o2 = generate_success(o, o2)
        #     replay_buffer.store(new_o, a, new_r * args.reward_scale, new_o2, d)

        # Store experience to replay buffer
        replay_buffer.store(o, a, get_reward(env) * args.reward_scale, o2, d)

        o = o2
        if d or (ep_len == args.max_ep_len):
            _, ep_ret, ep_len = env.reset(obs_as_dict=False), 0, 0
            o = get_observation(env)

        if t >= args.update_after and t % args.steps_per_update==0:
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
    parser.add_argument('--env', default='L2M2019Env', metavar='G',
                        help='name of environment name (default: L2M2019Env)')
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
    parser.add_argument('--difficulty', type=int, default=3, metavar='N',
                        help='difficulty for L2M2019Env(default: 3)')
    parser.add_argument('--max_ep_len', type=int, default=2500, metavar='N',
                        help='max_ep_len(default: 2500)')
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
                args.max_ep_len,    # max_ep_len
                args.epochs,        # total epochs
                4000,               # steps per epoch
                10000,              # start steps
                args.reward_scale,  # reward scale 
                4000,               # update after
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