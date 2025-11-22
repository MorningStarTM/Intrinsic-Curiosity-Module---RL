import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ICM.memory import Memory, GraphMemory
from ICM.Utils.logger import logger




class ICM(nn.Module):
    def __init__(self, config) -> None:
        super(ICM, self).__init__()
        self.config = config
        self.batch_size = self.config['batch_size']

        
        self.state = nn.Linear(self.config['input_dim'], 512)
        self.state_ = nn.Linear(self.config['input_dim'], 512)

        # inverse Model
        self.inverse_model = nn.Sequential(
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, self.config['action_dim'])
        )


        # forward model
        self.forward_model = nn.Sequential(
                        nn.Linear(513, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512)
        )


        self.optimizer = optim.Adam(self.parameters(), lr=self.config['icm_lr'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=self.device)
        self.memory = Memory()

    def _ensure_batch(self, x):
        # state embedding might be [512]; make it [1,512] for concat
        return x.unsqueeze(0) if x.dim() == 1 else x

    
    
    def forward(self, action, state, next_state):
        state = torch.tensor(state.to_vect(), dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state.to_vect(), dtype=torch.float, device=self.device)
        state = self.state(state)
        state_ = self.state_(next_state)

        action_ = self.inverse_model(torch.cat([state, state_], dim=-1))
        action_probs = F.softmax(action_, dim=-1)
        action_distribution = Categorical(action_probs)
        action_pred = action_distribution.sample()
        
        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)
        pred_next_state = self.forward_model(torch.cat([state, action], dim=-1))

        return state_, pred_next_state, action_pred, action_
    

    def calc_batch_loss(self, state_, pred_state, action_idx, action_logits):

        # add batch dim if single sample arrived
        if state_.dim() == 1:       state_ = state_.unsqueeze(0)
        if pred_state.dim() == 1:   pred_state = pred_state.unsqueeze(0)
        if action_logits.dim() == 1: action_logits = action_logits.unsqueeze(0)

        # ---- forward loss in feature space
        Lf = self.config['beta'] * F.mse_loss(pred_state, state_, reduction='mean')

        # ---- inverse loss (logits vs integer indices)  <-- NO unsqueeze, NO re-wrapping
        if isinstance(action_idx, int):
            action_idx = torch.tensor([action_idx], device=self.device, dtype=torch.long)
        else:
            action_idx = action_idx.to(self.device).long().view(-1)

        Li = (1.0 - self.config['beta']) * F.cross_entropy(action_logits, action_idx, reduction='mean')

        # ---- intrinsic reward (no grad), vector per sample

        return Li, Lf


    def calc_loss(self, state_, pred_state, action=None, action_pred=None):

        with torch.no_grad():
            intrinsic_reward = self.config['alpha'] * ((state_ - pred_state).pow(2)).mean(dim=0)
        return intrinsic_reward #Li, Lf
    

    def learn(self):
        states_, pred_states, actions, actions_pred = self.memory.sample_memory()

        states_ = torch.squeeze(torch.stack(states_, dim=0)).to(self.device)
        pred_states = torch.squeeze(torch.stack(pred_states, dim=0)).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1)
        actions_pred = torch.squeeze(torch.stack(actions_pred, dim=0)).to(self.device)

        # one shot loss & backward (no loops)
        Li, Lf = self.calc_batch_loss(states_, pred_states, actions, actions_pred)
        loss = Li + Lf

        #self.optimizer.zero_grad()
        #loss.backward()
        #self.optimizer.step()

        #logger.info(f"Average Loss: {float(loss.item()):.3f}")
        return loss


    def train(self):
        states_, pred_states, actions, actions_pred = self.memory.sample_memory()

        states_ = torch.squeeze(torch.stack(states_, dim=0)).to(self.device)
        pred_states = torch.squeeze(torch.stack(pred_states, dim=0)).to(self.device)
        #actions = torch.squeeze(torch.stack(actions, dim=0)).float().detach().to(self.device)
        actions = torch.stack([torch.tensor(a, dtype=torch.long) for a in actions], dim=0)
        actions_pred = torch.squeeze(torch.stack(actions_pred, dim=0)).to(self.device)

        logger.info(f"states : {states_.shape}, pred_states : {pred_states.shape}, actions : {actions.shape}, actions_pred : {actions_pred.shape}")

        # Initialize total loss
        total_loss = 0.0

        # Process data in batches
        num_records = states_.shape[0]
        for start_idx in range(0, num_records, self.config['batch_size']):
            # Define batch indices
            end_idx = start_idx + self.config['batch_size']
            state_batch = states_[start_idx:end_idx]
            pred_state_batch = pred_states[start_idx:end_idx]
            action_batch = actions[start_idx:end_idx]
            action_pred_batch = actions_pred[start_idx:end_idx]

            # Compute loss for the batch
            intrinsic_reward, Li, Lf = self.calc_batch_loss(
                state_batch, pred_state_batch, action_batch, action_pred_batch
            )
            batch_loss = Li + Lf
            print(Li, Lf)

            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += batch_loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / (num_records / self.config['batch_size'])
        logger.info(f"Average Loss: {avg_loss:.3f}")


    def save_checkpoint(self, filename="icm_checkpoint.pth"):
        os.makedirs(self.config['save_path'], exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        save_path = os.path.join(self.config['save_path'], filename)
        torch.save(checkpoint, save_path)
        logger.info(f"ICM model saved to {save_path}")

    def load_checkpoint(self, folder_name=None, filename="icm_checkpoint.pth"):
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join(self.config['save_path'], filename)
        if not os.path.exists(file_path):
            logger.error(f"[LOAD] No checkpoint found at {file_path}")
            return False
        
        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        logger.info(f"ICM model loaded from {filename}")






class GraphACICM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dim = config['action_dim']

        # encoders to 512-d feature space
        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        self.phi_s_n = nn.Linear(config['input_dim'], 512)

        # inverse: [phi(s) || phi(s')] -> a_logits
        self.inverse_model = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),  nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

        # forward: [phi(s) || one_hot(a)] -> phi_hat(s')
        self.forward_model = nn.Sequential(
            nn.Linear(512 + self.action_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 512)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=config['icm_lr'])
        self.memory = GraphMemory(capacity=config.get('replay_capacity', 50000))
        self.to(self.device)

    
    def forward(self, action_idx, state, next_state):
        """
        action_idx: int or 1D LongTensor [B]
        state, next_state: Grid2Op obs (scalar) or list/tuple of obs (batch)

        Returns:
            phi_s          : [B, 512]   encoded current features
            phi_s_next     : [B, 512]   encoded next features
            inv_logits     : [B, A]     inverse-model logits for action
            phi_hat_next   : [B, 512]   forward-model predicted next features
        """
        # ---- to batch ----
        if isinstance(state, (list, tuple)):
            s_list  = [self._to_tensor(o)  for o in state]
            sn_list = [self._to_tensor(o)  for o in next_state]
            s  = torch.stack(s_list,  dim=0)  # [B, F]
            sn = torch.stack(sn_list, dim=0)  # [B, F]
        else:
            s  = self._to_tensor(state).unsqueeze(0)      # [1, F]
            sn = self._to_tensor(next_state).unsqueeze(0) # [1, F]

        phi_s      = self.phi_s(s)        # [B,512]
        phi_s_next = self.phi_s_n(sn)     # [B,512]

        # ---- inverse head: predict action from (phi_s || phi_s_next)
        inv_logits = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=-1))  # [B,A]

        # ---- forward head: (phi_s || one_hot(a)) -> phi_hat_next
        if not torch.is_tensor(action_idx):
            action_idx = torch.tensor([int(action_idx)], dtype=torch.long, device=self.device)  # [1]
        else:
            action_idx = action_idx.to(self.device).view(-1)  # [B]

        a_oh = F.one_hot(action_idx, num_classes=self.action_dim).float()        # [B,A]
        phi_hat_next = self.forward_model(torch.cat([phi_s, a_oh], dim=-1))      # [B,512]

        return phi_s, phi_s_next, inv_logits, phi_hat_next

    # ---------- helpers ----------
    def _to_tensor_vec(self, v):
        v = torch.as_tensor(v, dtype=torch.float32, device=self.device)
        return v

    def encode_batch(self, obs_vec_list):
        x = torch.stack([self._to_tensor_vec(v) for v in obs_vec_list], dim=0)  # [B,F]
        return self.encoder(x)  # [B,512]
    
    def encode(self, obs):
        """
        Accepts a Grid2Op obs or a precomputed vector; returns φ(s) as [512].
        """
        vec = obs.to_vect() if hasattr(obs, "to_vect") else obs
        x = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,F]
        return self.encoder(x).squeeze(0)  # [512]

    def encode_next(self, obs_next):
        # Using the same shared encoder; kept for backwards compatibility.
        return self.encode(obs_next)

    def action_onehot_batch(self, idx_tensor):
        return F.one_hot(idx_tensor, num_classes=self.action_dim).float()

    def intrinsic(self, phi_next, phi_hat_next):
        # 0.5 * ||·||^2 is common; keep alpha as scale
        return self.config['alpha'] * 0.5 * F.mse_loss(phi_hat_next, phi_next, reduction='none').mean()

    # ---------- learning step ----------
    def _to_index_tensor(self, a_list):
        idx = [int(a.detach().view(-1)[0].item()) if torch.is_tensor(a) else int(a) for a in a_list]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def learn(self):
        obs_list, obs_next_list, a_list = self.memory.sample_memory(self.config.get('batch_size'))
        if len(obs_list) == 0:
            return torch.zeros((), device=self.device)

        # Re-encode with CURRENT encoder (grads flow into encoder)
        phi_s      = self.encode_batch(obs_list)       # [B,512]
        phi_s_next = self.encode_batch(obs_next_list)  # [B,512]
        a_idx      = self._to_index_tensor(a_list)     # [B]
        a_oh       = self.action_onehot_batch(a_idx)   # [B,A]

        # inverse loss
        inv_logits = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=-1))  # [B,A]
        Li = (1.0 - self.config['beta']) * F.cross_entropy(inv_logits, a_idx)

        # forward loss
        phi_hat_next = self.forward_model(torch.cat([phi_s, a_oh], dim=-1))      # [B,512]
        Lf = self.config['beta'] * F.mse_loss(phi_hat_next, phi_s_next)

        return Li + Lf
