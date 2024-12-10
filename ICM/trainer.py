import grid2op
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend
from grid2op import Environment
from ICM.converter import ActionConverter
import torch.optim as optim
from grid2op.Exceptions import *
from tqdm import tqdm
from ICM.Utils.logger import logging


class Trainer:
    def __init__(self, agent, env:Environment, converter:ActionConverter, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.converter = converter
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.config['lr'], betas=self.config['betas'])

    
    def train(self):
        num_episodes = len(self.env.chronics_handler.subpaths)

        for episode_id in range(num_episodes):

            print(f"Episode ID : {episode_id}")
            self.env.set_id(episode_id)
            obs = self.env.reset()
            reward = self.env.reward_range[0]
            done = False

            for i in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode_id}", leave=True):
                
                try:
                    action = self.agent(obs.to_vect()) 
                    obs_, reward, done, _ = self.env.step(self.converter.act(action))
                    self.agent.rewards.append(reward)
                    obs = obs_

                    if done:
                        self.env.set_id(episode_id)
                        
                        obs = self.env.reset()
                        done = False
                        reward = self.env.reward_range[0]

                        self.env.fast_forward_chronics(i - 1)
                        action = self.agent(obs.to_vect()) 
                        obs_, reward, done, _ = self.env.step(self.converter.act(action))
                        self.agent.rewards.append(reward)


                except NoForecastAvailable as e:
                    logging.info(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue

                except Grid2OpException as e:
                    logging.info(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue 