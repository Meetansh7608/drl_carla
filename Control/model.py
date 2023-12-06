import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class ConvNet(nn.Module):
    def __init__(self, dim, in_channels, num_actions) -> None:
        super(ConvNet, self).__init__()
        # conv2d(in_channels, out_channels, kernel_size, stride)
        
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.fc1_bn(self.fc1(x.reshape(-1, 64*8*8))))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(
        self,
        num_actions,
        state_dim, #?
        in_channels,
        device,
        discount=0.9,
        optimizer="Adam",
        optimizer_parameters={'lr':0.01},
        target_update_frequency=1e4,
        initial_eps = 1,
        end_eps = 0.05,
        eps_decay_period = 25e4,
        eval_eps=0.001
    ) -> None:
        self.device = device

        self.Q = ConvNet(state_dim, in_channels, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)  # copy target network
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), 
        **optimizer_parameters)

        self.discount = discount

        self.target_update_frequency = target_update_frequency
 
        # epsilon decay
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        self.state_shape = (-1,) + state_dim
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
        else max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        self.current_eps = eps

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > eps:
            self.Q.eval()
            with torch.no_grad():
                # without batch norm, remove the unsqueeze
                state = torch.FloatTensor(state).reshape(self.state_shape).unsqueeze(0).to(self.device)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def train(self, replay_buffer):
        self.Q.train()
        # Sample mininbatch from replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + (1-done) * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

        # Get current Q estimate
        # torch gather just selects action values from Q(state) using the action tensor as an index
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss
        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by full copy every X iterations.
        self.iterations += 1
        self.copy_target_update()
    
    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            print('target network updated')
            print('current epsilon', self.current_eps)
            self.Q_target.load_state_dict(self.Q.state_dict())


    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


    def load(self, filename):
        print(filename, "\n\n\n\n")
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
    
class Actor(nn.Module):
    def __init__(self, dim, in_channels, num_actions) -> None:
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.fc1_bn(self.fc1(x.reshape(-1, 64 * 8 * 8))))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = torch.tanh(self.fc3(x))  # Using tanh for actor output
        return x

class Critic(nn.Module):
    def __init__(self, dim, in_channels, num_actions) -> None:
        super(Critic, self).__init__()

        # Critic takes both state and action as input
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 8 * 8 + num_actions, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        x_state = F.relu(self.conv1_bn(self.conv1(state)))
        x_state = F.relu(self.conv2_bn(self.conv2(x_state)))
        x_state = F.relu(self.conv3_bn(self.conv3(x_state)))
        x_state = F.relu(self.fc1_bn(self.fc1(x_state.reshape(-1, 64 * 8 * 8))))

        x = torch.cat((x_state, action), dim=1)  # Concatenate state and action
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

class DDPG(object):
    def __init__(
        self,
        num_actions,
        state_dim,
        in_channels,
        device,
        discount=0.9,
        actor_optimizer="Adam",
        critic_optimizer="Adam",
        optimizer_parameters={'lr': 0.01},
        target_update_frequency=1e4,
        initial_eps=1,
        end_eps=0.05,
        eps_decay_period=25e4,
        eval_eps=0.001,
        tau=0.005  # Soft update parameter for target networks
    ) -> None:
        self.device = device
        self.tau = tau

        self.actor = Actor(state_dim, in_channels, num_actions).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, in_channels, num_actions).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = getattr(torch.optim, actor_optimizer)(self.actor.parameters(), **optimizer_parameters)
        self.critic_optimizer = getattr(torch.optim, critic_optimizer)(self.critic.parameters(), **optimizer_parameters)

        self.discount = discount
        self.target_update_frequency = target_update_frequency

        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (end_eps - initial_eps) / eps_decay_period

        self.state_shape = (-1,) + state_dim
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        self.current_eps = eps

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0, 1) > eps:
            self.actor.eval()
            with torch.no_grad():
                # without batch norm, remove the unsqueeze
                state = torch.FloatTensor(state).reshape(self.state_shape).unsqueeze(0).to(self.device)
                return self.actor(state).cpu().numpy().squeeze()
        else:
            return np.random.uniform(-1, 1, self.num_actions)

    def train(self, replay_buffer):
        self.actor.train()
        self.critic.train()

        # Sample mininbatch from replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Convert to PyTorch tensors
        state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).reshape(self.state_shape).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Compute the target Q value
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = reward + (1 - done) * self.discount * self.critic_target(next_state, next_action)

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss (policy gradient)
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.c)