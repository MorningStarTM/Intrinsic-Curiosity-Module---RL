import random
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
    





class GraphMemory:
    """
    Replay for ICM:
      - Stores tuples (obs_vect, next_obs_vect, action_idx)
      - obs_vect are plain float arrays from obs.to_vect()
    """
    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self.obs      = deque(maxlen=capacity)
        self.obs_next = deque(maxlen=capacity)
        self.actions  = deque(maxlen=capacity)

    def __len__(self): return len(self.actions)

    def _to_int(self, a):
        if hasattr(a, "detach"): return int(a.detach().view(-1)[0].item())
        return int(a)

    # new signature only: remember(obs, next_obs, action_idx)
    def remember(self, obs, next_obs, action_idx):
        self.obs.append(obs.to_vect() if hasattr(obs, "to_vect") else obs)
        self.obs_next.append(next_obs.to_vect() if hasattr(next_obs, "to_vect") else next_obs)
        self.actions.append(self._to_int(action_idx))

    def clear_memory(self):
        self.obs.clear(); self.obs_next.clear(); self.actions.clear()

    def sample_memory(self, batch_size: Optional[int] = None):
        n = len(self.actions)
        if n == 0: return [], [], []
        if batch_size is None or batch_size >= n:
            return list(self.obs), list(self.obs_next), list(self.actions)
        idxs = random.sample(range(n), batch_size)
        return ([self.obs[i] for i in idxs],
                [self.obs_next[i] for i in idxs],
                [self.actions[i] for i in idxs])

