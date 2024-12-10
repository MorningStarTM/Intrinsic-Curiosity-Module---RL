import torch 
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import grid2op
from ICM.converter import ActionConverter
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import gym
import os


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = self.config['input_dim']

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.policy = nn.Linear(256, self.config['action_dim'])
        self.value = nn.Linear(256, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []


    def forward(self, state):
        state = torch.tensor(state, device=self.device)
        state = F.relu(self.network(state))

        state_value = self.value(state)

        action_probs = F.softmax(self.policy(state), dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()


    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


    def save_model(self, model_name="actor_critic"):
        torch.save(self.state_dict(), os.path.join(self.config['save_path'], model_name))
        print(f"model saved at {os.path.join(self.config['save_path'])}")
        
    def load_model(self, model_name="actor_critic"):
        self.load_state_dict(torch.load(os.path.join(self.config['save_path'], model_name)))
        print(f"Model loaded from {os.path.join(self.config['save_path'])}")

        

