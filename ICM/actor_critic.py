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
from ICM.Utils.logger import logger
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, global_mean_pool


class ActorCriticGAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_dim   = config['input_dim']      # 11
        act_dim  = config['action_dim']     # size of your action catalog
        edge_dim = 5  # 5 (rho, p, q, status, dir)
        hid      = 256
        heads    = 4
        p_drop   = 0.0
        attn_drop= 0.0

        # Keep it shallow: 2 GATv2 layers
        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hid // heads,
            heads=heads,
            concat=True,               # -> [N, hid]
            edge_dim=edge_dim,         # <— use edge_attr
            dropout=attn_drop,
            add_self_loops=False,      # we don't add self-loops here to avoid attr mismatch
        )
        self.gat2 = GATv2Conv(
            in_channels=hid,
            out_channels=hid,
            heads=1,
            concat=False,              # -> [N, hid]
            edge_dim=edge_dim,
            dropout=attn_drop,
            add_self_loops=False,
        )

        self.act = nn.ReLU()
        self.do  = nn.Dropout(p_drop)

        # Graph-level heads (one action/value per grid)
        self.policy = nn.Linear(hid, act_dim)
        self.value  = nn.Linear(hid, 1)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'], betas=config['betas'])

        # (optional) rollout lists
        self.logprobs, self.state_values, self.rewards = [], [], []

    def forward(self, x, edge_index, batch, edge_attr):
        h = self.gat1(x, edge_index, edge_attr); h = self.act(h); h = self.do(h)
        h = self.gat2(h, edge_index, edge_attr); h = self.act(h)
        g = global_mean_pool(h, batch)           # [B, hid]
        logits = self.policy(g)                  # [B, action_dim]
        value  = self.value(g).squeeze(-1)       # [B]
        
        action_probs = F.softmax(logits, dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(value)

        return action.item()
    

    def calculateLoss(self, gamma=0.99):
        if not (self.logprobs and self.state_values and self.rewards):
            logger.error("Warning: Empty memory buffers!")
            return torch.tensor(0.0, device=self.device)
        

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
       
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward.unsqueeze(0))
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def save_checkpoint(self, filename="graph_actor_critic_checkpoint.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs(self.config['save_path'], exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        save_path = os.path.join(self.config['save_path'], filename)
        torch.save(checkpoint, save_path)
        logger.info(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="graph_actor_critic_checkpoint.pth", load_optimizer=True):
        """Load model + optimizer state."""
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join(self.config['save_path'], filename)
        if not os.path.exists(file_path):
            logger.error(f"[LOAD] No checkpoint found at {file_path}")
            return False

        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"[LOAD] Checkpoint loaded from {file_path}")
        return True
    


        


    




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

        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'], betas=self.config['betas'])
        self.to(self.device)


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
        if not (self.logprobs and self.state_values and self.rewards):
            logger.error("Warning: Empty memory buffers!")
            return torch.tensor(0.0, device=self.device)
        

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
       
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward.unsqueeze(0))
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def save_checkpoint(self, filename="actor_critic_checkpoint.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs(self.config['save_path'], exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        save_path = os.path.join(self.config['save_path'], filename)
        torch.save(checkpoint, save_path)
        logger.info(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="actor_critic_checkpoint.pth", load_optimizer=True):
        """Load model + optimizer state."""
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join(self.config['save_path'], filename)
        if not os.path.exists(file_path):
            logger.error(f"[LOAD] No checkpoint found at {file_path}")
            return False

        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"[LOAD] Checkpoint loaded from {file_path}")
        return True
    


        

