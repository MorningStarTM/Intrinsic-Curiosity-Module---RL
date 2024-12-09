import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import grid2op
from converter import ActionConverter
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import gym


class SharedAdam(T.optim.Adam):
    def __init__(self, params, config:dict):
        super(SharedAdam, self).__init__(params, lr=config['lr'], betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay'])

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()




class ActorCritic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.gamma = config['gamma']
        self.config = config
        self.f1 = nn.Linear(self.config['input_dim'], 512)
        self.f2 = nn.Linear(512, 1024)
        self.f3 = nn.Linear(1024, 512)
        self.f4 = nn.Linear(512, 256)
        self.pi = nn.Linear(256, self.config['action_dim'])
        self.v = nn.Linear(256, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []


    def forward(self, state):
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f4(x))

        pi = self.pi(x)
        v = self.v(x)

        return pi, v
    

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1] * (1-int(done))

        batch_return = []

        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return
    

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        valuess = values.squeeze()
        critic_loss = (returns - values)**2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss
    

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]
        return action



class Agent(mp.Process):
    def __init__(self, global_actor_critic, env, converter:ActionConverter, optimizer:SharedAdam, name, config):
        super(Agent, self).__init__()
        self.config = config
        self.local_actor_critic = ActorCritic(config=config)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = self.config['global_ep_idx']
        self.env = env
        self.optimizer = optimizer
        self.converter = converter

    def run(self):
        t_step = 1
        print("STARTING --------------------------------")
        while self.config['global_ep_idx'].value < self.config['N_GAMES']:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation.to_vect())
                observation_, reward, done, info = self.env.step(self.converter.act(action))
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % self.config['T_MAX'] == 0 or done:
                    print("TRAINING------------------------------")
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()

                    for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                                         self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_

            with self.config['global_ep_idx'].get_lock():
                self.config['global_ep_idx'].value += 1
            print(self.config['env_name'], 'episode', self.config['global_ep_idx'].value, 'reward %.1f' % score)

            


if __name__ == "__main__":
    config = {}
    config['gamma'] = 0.99
    config['eps'] = float(1e-8)
    config['betas'] = (0.9, 0.99)
    config['weight_decay'] = 0
    config['lr'] = 0.0001
    config['env_name'] = 'l2rpn_case14_sandbox'
    config['action_dim'] = 179
    config['N_GAMES'] = 3000
    config['input_dim'] = 467
    config['T_MAX'] = 5
    config['global_ep_idx'] = mp.Value('i', 0)


    env = grid2op.make(config['env_name'], reward_class=L2RPNSandBoxScore,
                                backend=LightSimBackend())


    global_actor_critic = ActorCritic(config=config)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), config=config)

    
    converter = ActionConverter(env=env)
    
    workers = [Agent(global_actor_critic, env=env, converter=converter, optimizer=optim, name=i, config=config) for i in range(mp.cpu_count())]

    [w.start() for w in workers]
    [w.join() for w in workers]