import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import gym
import random
import collections


class Actor(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        ret = F.relu(self.Linear1(states))
        return F.softmax(self.Linear2(ret), dim=1)


class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        ret = F.relu(self.Linear1(states))
        return self.Linear2(ret)


class SAC:

    def __init__(self, state_dim, hidden_dim, action_dim, target_entropy, tau, gamma):
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.critic1 = Critic(state_dim, hidden_dim, action_dim)
        self.critic1_target = Critic(state_dim, hidden_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2 = Critic(state_dim, hidden_dim, action_dim)
        self.critic2_target = Critic(state_dim, hidden_dim, action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau

    def calc_td_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.critic1_target(next_states)
        q2_value = self.critic2_target(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def update(self, data):
        state, action, reward, next_state, done = zip(*data)
        states = torch.tensor(state, dtype=torch.float)
        actions = torch.tensor(action, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(reward, dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(next_state, dtype=torch.float)
        dones = torch.tensor(done, dtype=torch.long).squeeze().view(-1, 1)
        td_target = self.calc_td_target(rewards, next_states, dones)
        critic1_q_values = self.critic1(states).gather(1, actions)
        critic1_loss = torch.mean(F.mse_loss(critic1_q_values, td_target.detach()))
        critic2_q_values = self.critic2(states).gather(1, actions)
        critic2_loss = torch.mean(F.mse_loss(critic2_q_values, td_target.detach()))
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)

        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)


    def take_action(self, state):
        # print(state)
        state = torch.tensor([state], dtype=torch.float)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        data = random.sample(self.buffer, batch_size)
        return data

    def size(self):
        return len(self.buffer)

actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 500
batch_size = 128
target_entropy = -1
env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = SAC(state_dim, hidden_dim, action_dim, target_entropy, tau, gamma)

for i in range(num_episodes):
    state = env.reset()
    done = 0
    G = 0
    while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        if replay_buffer.size() > minimal_size:
            data = replay_buffer.sample(batch_size)
            agent.update(data)
        G += reward
    if i % 20 == 0:
        # print(i, " ", G)
        print('Episode ', '%6d' % i,  ' tot_reward: ', '%10f' %G, )