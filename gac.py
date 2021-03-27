import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from copy import deepcopy
import itertools
import numpy as np

from utils import MLPActorCritic,mmd

class GAC(object):
    def __init__(self, 
                actor_critic, 
                replay_buffer,
                lr_actor =1e-3,
                lr_critic=1e-3,
                lr_target_actor =1.0,
                lr_target_critic=0.005,
                expand_batch=100,
                gamma=0.99,
                alpha_start=1.2, alpha_min=1.0, alpha_max=1.5, alpha_lr=1e-3,
                beta_start=0.3, beta_min=0.3, beta_max=4.0, beta_lr=0.01,
                kernel='energy',
                device=torch.device("cuda:0")):
        self.actor_critic  = deepcopy(actor_critic)
        self.replay_buffer = replay_buffer
        self.target_actor_critic = actor_critic
        self.optim_actor  = Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.critic_param = itertools.chain(self.actor_critic.critic1.parameters(), self.actor_critic.critic2.parameters())
        self.optim_critic = Adam(self.critic_param, lr=lr_critic)
        self.lr_target_actor = lr_target_actor
        self.lr_target_critic = lr_target_critic
        self.expand_batch = expand_batch
        self.gamma = gamma
        self.alpha, self.alpha_min, self.alpha_max = alpha_start, alpha_min, alpha_max
        self.beta, self.beta_min, self.beta_max, self.beta_lr  = beta_start, beta_min, beta_max, beta_lr
        self.kernel = kernel
        self.device = device

        self.log_alpha = torch.tensor(np.log(alpha_start), requires_grad=True, device=device)
        self.optim_log_alpha = Adam([self.log_alpha], lr=alpha_lr)

        for p in self.target_actor_critic.parameters():
            p.requires_grad = False
    
    def update_obs_param(self):
        # for state normalization
        self.actor_critic.obs_mean = torch.FloatTensor(self.replay_buffer.obs_mean).to(self.device)
        self.actor_critic.obs_std  = torch.FloatTensor(self.replay_buffer.obs_std).to(self.device)
        self.target_actor_critic.obs_mean = self.actor_critic.obs_mean
        self.target_actor_critic.obs_std  = self.actor_critic.obs_std

    # def update_obs_param2(self):
    #     # for state normalization
    #     self.replay_buffer.obs_mean = self.actor_critic.obs_mean.cpu().numpy()
    #     self.replay_buffer.obs_std = self.actor_critic.obs_std.cpu().numpy()
    #     self.replay_buffer.obs_square_mean = np.maximum(self.replay_buffer.obs_std**2 + self.replay_buffer.obs_mean**2, 1e-8)
    #     self.target_actor_critic.obs_mean = self.actor_critic.obs_mean
    #     self.target_actor_critic.obs_std  = self.actor_critic.obs_std

    def update(self, batch_size):
        if isinstance(self.replay_buffer, list):
            length = len(self.replay_buffer)
            idx = np.random.randint(length)
            data = self.replay_buffer[idx].sample_batch(batch_size) # normalized by replay buffer
        else:
            data = self.replay_buffer.sample_batch(batch_size) # normalized by replay buffer
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        o  = torch.FloatTensor(o).to(self.device)
        a  = torch.FloatTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        o2 = torch.FloatTensor(o2).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        self.optim_critic.zero_grad()
        loss_c = self.loss_critic(o, a, r, o2, d)
        loss_c.backward()
        self.optim_critic.step()

        for p in self.critic_param:
            p.requires_grad = False
        self.optim_actor.zero_grad()
        loss_a, mmd_entropy = self.loss_actor(o)
        loss_a.backward()
        self.optim_actor.step()
        for p in self.critic_param:
            p.requires_grad = True

        self.optim_log_alpha.zero_grad()
        loss = self.loss_log_alpha(mmd_entropy)
        loss.backward()
        self.optim_log_alpha.step()
        self.alpha = torch.exp(self.log_alpha).detach()

        self.update_target_actor()
        self.update_target_critic()

        return loss_a.detach().cpu().numpy(), loss_c.detach().cpu().numpy(), self.alpha.cpu().numpy()

    def loss_actor(self, o):    
        o2 = o.repeat(self.expand_batch, 1)
        a2 = self.actor_critic.actor(o2)

        q1  = self.actor_critic.critic1(o2, a2)
        q2  = self.actor_critic.critic2(o2, a2)

        a2 = a2.view(self.expand_batch, -1, a2.shape[-1]).transpose(0, 1)
        with torch.no_grad():
            a3 = (2 * torch.rand_like(a2) - 1)

        mmd_entropy = mmd(a2, a3, kernel=self.kernel)

        # Entropy-regularized policy loss
        loss = -torch.min(q1, q2).mean() + self.alpha * mmd_entropy
        return loss, mmd_entropy.detach()

    def loss_log_alpha(self, mmd_entropy):
        if self.log_alpha < -5.0:
            loss = -self.log_alpha
        elif self.log_alpha > 5.0:
            loss = self.log_alpha
        else:
            loss = self.log_alpha * (self.beta - mmd_entropy)
        return loss

    def loss_critic(self, o, a, r, o2, d):
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2 = self.target_actor_critic.actor(o2)

            # Target Q-values
            target_q1 = self.target_actor_critic.critic1(o2, a2)
            target_q2 = self.target_actor_critic.critic2(o2, a2)
            backup = r + self.gamma * (1 - d) * torch.min(target_q1, target_q2)

        # MSE loss against Bellman backup
        q1 = self.actor_critic.critic1(o, a)
        q2 = self.actor_critic.critic2(o, a)
        loss = ((q1 - backup)**2).mean() + ((q2 - backup)**2).mean() 
        return loss
    
    def update_target_actor(self):
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.actor.parameters(), self.target_actor_critic.actor.parameters()):
                p_targ.data.mul_((1 - self.lr_target_actor))
                p_targ.data.add_(self.lr_target_actor * p.data)

    def update_target_critic(self):
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.critic1.parameters(), self.target_actor_critic.critic1.parameters()):
                p_targ.data.mul_((1 - self.lr_target_critic))
                p_targ.data.add_(self.lr_target_critic * p.data)
            for p, p_targ in zip(self.actor_critic.critic2.parameters(), self.target_actor_critic.critic2.parameters()):
                p_targ.data.mul_((1 - self.lr_target_critic))
                p_targ.data.add_(self.lr_target_critic * p.data)
    
    def update_beta(self):
        if self.alpha > self.alpha_max:
            self.beta += self.beta_lr
            self.beta = min(self.beta, self.beta_max)
        elif self.alpha < self.alpha_min:
            self.beta -= self.beta_lr
            self.beta = max(self.beta, self.beta_min)