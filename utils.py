import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for GAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, obs_limit=5.0):
        self.obs_buf  = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

# ==============================================================================
# Model For Actor Critic

def _weight_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)

class GenerativeGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.epsilon_dim = act_dim * act_dim
        hidden_sizes[0] += self.epsilon_dim
        self.net = mlp([obs_dim+self.epsilon_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh())
        self.apply(_weight_init)

    def forward(self, obs, std=1.0, noise='gaussian', epsilon_limit=5.0):
        if noise == 'gaussian':
            # Gaussian Distribution
            epsilon = (std * torch.randn(obs.shape[0], self.epsilon_dim, device=obs.device)).clamp(-epsilon_limit, epsilon_limit)
        else:
            # Uniform Distribution
            epsilon = torch.rand(obs.shape[0], self.epsilon_dim, device=obs.device) * 2 - 1
        pi_action = self.net(torch.cat([obs, epsilon], dim=-1))
        return pi_action

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        hidden_sizes[0] += act_dim
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.apply(_weight_init)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q = self.q(x)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(400,300),
                 activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        # build policy and value functions
        self.actor = GenerativeGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.critic1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.critic2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False, noise='gaussian', obs_limit=5.0):
        with torch.no_grad():
            if deterministic:
                a = self.actor(obs, std=0.5, noise=noise)
            else:
                a = self.actor(obs, noise=noise)
        return a.detach().cpu().numpy()[0]

#===============================================================================

# ==============================================================================
# Maximum Mean Discrepancy
# geomloss: https://github.com/jeanfeydy/geomloss

class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    return Sqrt0.apply(x)

def squared_distances(x, y):
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy

def gaussian_kernel(x, y, blur=1.0):
    C2 = squared_distances(x / blur, y / blur)
    return (- .5 * C2 ).exp()

def energy_kernel(x, y, blur=None):
    return -squared_distances(x, y)

kernel_routines = {
    "gaussian" : gaussian_kernel,
    "energy"   : energy_kernel,
}

def mmd(x, y, kernel='gaussian'):
    b = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]

    if kernel in kernel_routines:
        kernel = kernel_routines[kernel]

    K_xx = kernel(x, x).mean()
    K_xy = kernel(x, y).mean()
    K_yy = kernel(y, y).mean()

    return sqrt_0(K_xx + K_yy - 2*K_xy)

# ==============================================================================