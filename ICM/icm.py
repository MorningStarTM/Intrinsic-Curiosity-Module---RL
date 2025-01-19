import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ICM.memory import Memory

class ICM(nn.Module):
    def __init__(self, config) -> None:
        super(ICM, self).__init__()
        self.config = config

        
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

    def forward(self, action, state, next_state):
        state = self.state(state)
        state_ = self.state_(next_state)

        action_ = self.inverse_model(torch.cat([state, state_], dim=-1))
        action_probs = F.softmax(action_, dim=-1)
        action_distribution = Categorical(action_probs)
        action_pred = action_distribution.sample()

        pred_next_state = self.forward_model(torch.cat([state, action], dim=-1))

        return state_, pred_next_state, action_pred, action_
    

    def calc_loss(self, state_, pred_state, action, action_pred):
        inverse_loss = nn.MSELoss()
        Lf = inverse_loss(state_, pred_state)
        Lf = self.config['beta'] * Lf

        forward_loss = nn.CrossEntropyLoss()
        action_pred = action_pred.unsqueeze(0).float()
        act = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)
        print(f"action pred : {action_pred.shape}")
        print(f"action : {act.shape}")
        Li = forward_loss(action_pred, act)
        Li = (1-self.config['beta']) * Li

        intrinsic_reward = self.config['alpha'] * ((state_ - pred_state).pow(2)).mean(dim=0)
        return intrinsic_reward, Li, Lf
    


    def learn(self, state_, pred_state, action, action_pred, batch_size=32):
        # Convert inputs to tensors
        state_ = torch.tensor(state_, dtype=torch.float, device=self.device)
        pred_state = torch.tensor(pred_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        action_pred = torch.tensor(action_pred, dtype=torch.float, device=self.device)

        # Initialize total loss
        total_loss = 0.0

        # Process data in batches
        num_records = state_.shape[0]
        for start_idx in range(0, num_records, batch_size):
            # Define batch indices
            end_idx = start_idx + batch_size
            state_batch = state_[start_idx:end_idx]
            pred_state_batch = pred_state[start_idx:end_idx]
            action_batch = action[start_idx:end_idx]
            action_pred_batch = action_pred[start_idx:end_idx]

            # Compute loss for the batch
            intrinsic_reward, Li, Lf = self.calc_loss(
                state_batch, pred_state_batch, action_batch, action_pred_batch
            )
            batch_loss = Li + Lf

            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += batch_loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / (num_records / batch_size)
        print(f"Average Loss: {avg_loss:.3f}")









