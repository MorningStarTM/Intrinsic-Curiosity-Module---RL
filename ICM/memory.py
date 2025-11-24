import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class Memory:
    def __init__(self):
        self.states_ = []
        self.pred_states = []
        self.actions_pred = []
        self.actions = []

    def remember(self, state_, pred_state, actions, pred_actions):
        self.actions.append(actions)
        self.states_.append(state_)
        self.pred_states.append(pred_state)
        self.actions_pred.append(pred_actions)

    def clear_memory(self):
        self.states_ = []
        self.pred_states = []
        self.actions_pred = []
        self.actions = []

    def sample_memory(self):
        return self.states_, self.pred_states, self.actions, self.actions_pred
    





