import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ICM(nn.Module):
    def __init__(self, config) -> None:
        super(ICM).__init__()
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


    def forward(self, action, state, next_state):
        state = self.state(state)
        state_ = self.state_(next_state)

        action_pred = self.inverse_model(torch.cat([state, state_], dim=-1))

        pred_next_state = self.forward_model(torch.cat([state, action], dim=-1))

        return state_, pred_next_state, action_pred
    

    def calc_loss(self, state_, pred_state, action, action_pred):
        Lf = nn.MSELoss(state_, pred_state)
        Lf = self.config['beta'] * Lf

        Li = nn.CrossEntropyLoss(action, action_pred)
        Li = (1-self.config['beta']) * Li

        intrinsic_reward = self.config['alpha'] * ((state_, pred_state).pow(2)).mean(dim=1)
        return intrinsic_reward, Li, Lf
    

    


    

