import torch 
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import grid2op
from ICM.converter import ActionConverter
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import os
from ICM.Utils.logger import logger
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops


class ActorCriticGAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        in_dim   = config['input_dim']      # 11
        act_dim  = config['action_dim']     # size of your action catalog
        edge_dim = 5  # 5 (rho, p, q, status, dir)
        p_drop   = 0.0
        attn_drop= 0.0

        # shared trunk: 64 -> 128 -> 256
        self.g1 = GATv2Conv(in_dim, 16, heads=4, concat=True,  add_self_loops=True)   # 64
        self.g2 = GATv2Conv(64, 32, heads=4, concat=True,  add_self_loops=True)   # 128
        self.g3 = GATv2Conv(128, 256, heads=1, concat=False, add_self_loops=True)   # 256

        # policy head IS a graph layer: 256 -> act_dim (per node)
        self.gp = GATv2Conv(256, act_dim, heads=1, concat=False, add_self_loops=True) # [N, A]

        # value head IS a graph layer: 256 -> 1 (per node)
        self.gv = GATv2Conv(256, 1, heads=1, concat=False, add_self_loops=True)       # [N, 1]

        self.act = nn.ReLU()
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'], betas=config['betas'])

        self.logprobs, self.state_values, self.rewards = [], [], []

    def forward(self, x, edge_index, batch, _edge_attr_ignored=None):
        x = torch.nan_to_num(x, 0.0, 1e6, -1e6)

        h = self.act(self.g1(x, edge_index))
        h = self.act(self.g2(h, edge_index))
        h = self.act(self.g3(h, edge_index))          # [N,256]

        # per-node action scores, then POOL to graph logits
        pa = self.gp(h, edge_index)                   # [N, A]
        va = self.gv(h, edge_index)                   # [N, 1]

        logits = global_add_pool(pa, batch)           # [B, A]  (sum avoids NaN)
        value  = global_add_pool(va, batch).squeeze(-1)  # [B]

        logits = torch.nan_to_num(logits, 0.0, 1e6, -1e6)
        dist = Categorical(logits=logits)
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action))   # [B] (B=1 in your loop)
        self.state_values.append(value)               # [B]
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
        self.logprobs.clear()
        self.state_values.clear()
        self.rewards.clear()

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
        super().__init__()
        self.config = config
        self.affine = nn.Sequential(
            nn.Linear(self.config['input_dim'], 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),                 # ← normalize hidden to tame scales
            nn.ReLU(),
        )
        self.action_layer = nn.Linear(256, self.config['action_dim'])
        self.value_layer  = nn.Linear(256, 1)

        self.logprobs, self.state_values, self.rewards = [], [], []

        # Optional: safer initializations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def _sanitize(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x.clamp_(-1e6, 1e6)
        return x

    def forward(self, state_np):
        x = torch.from_numpy(state_np).float().to(self.value_layer.weight.device)
        x = self._sanitize(x)

        h = self.affine(x)                       # includes LayerNorm + ReLU
        h = torch.nan_to_num(h)                  # belt & suspenders

        logits = self.action_layer(h)
        logits = torch.nan_to_num(logits)        # if any NaN slipped through
        logits = logits - logits.max()           # stable softmax
        probs  = torch.softmax(logits, dim=-1)

        # final guard
        if not torch.isfinite(probs).all():
            # Zero-out non-finites and renormalize as an emergency fallback
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            s = probs.sum()
            probs = (probs + 1e-12) / (s + 1e-12)

        dist   = Categorical(probs=probs)
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action))
        self.state_values.append(self.value_layer(h).squeeze(-1))

        return action.item()


    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards, device=self.device)
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

    def save_checkpoint(self, optimizer:optim, filename="actor_critic_checkpoint.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs(self.config['save_path'], exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }
        save_path = os.path.join(self.config['save_path'], filename)
        torch.save(checkpoint, save_path)
        logger.info(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="actor_critic_checkpoint.pth", optimizer=None, load_optimizer=True):
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"[LOAD] Checkpoint loaded from {file_path}")
        return True
    


        

