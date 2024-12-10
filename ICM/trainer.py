import grid2op
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend
from grid2op import Environment
from ICM.converter import ActionConverter
from ICM.actor_critic import ActorCritic
import torch.optim as optim
from grid2op.Exceptions import *
from tqdm import tqdm
from ICM.Utils.logger import logging
import random
import inspect


class Trainer:
    def __init__(self, agent:ActorCritic, env:Environment, converter:ActionConverter, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.converter = converter
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.config['lr'], betas=self.config['betas'])
        self.best_survival_step = 0

    def train(self):
        num_episodes = len(self.env.chronics_handler.subpaths)

        for episode_id in range(num_episodes-100):

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


            if episode_id!=0 and episode_id % 5 == 0:
                print(f"\n\n#############################################\n\nEvaluating the Agent\n\n#############################################\n\n")
                num_steps, rewards = self.evaluate()
                if self.best_survival_step < num_steps:
                    print(f"Agent survived {num_steps}/{self.env.max_episode_duration()} steps")
                    self.agent.save_model(model_name=f"actor_critic_{num_steps}")
                    self.best_survival_step = num_steps



    def evaluate(self):
        num_steps = 0
        rewards = []
        paths = self.env.chronics_handler.subpaths
        test_path = random.choice(paths[900:])
        logging.info(f"Selected Chronics : {test_path}")

        try:
            self.env.set_id(test_path)
            logging.info(f"Selected Chronic loaded")
        except Exception as e:
            print("Error occured {e}")
            logging(f"{self.__class__.__name__}.{__name__} Error Occured")

        obs = self.env.reset()
        reward = self.env.reward_range[0]
        done = False

        for i in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {test_path}", leave=True):
            num_steps += i
            try:
                action = self.agent(obs.to_vect()) 
                obs_, reward, done, _ = self.env.step(self.converter.act(action))
                rewards.append(reward)
                obs = obs_

                if done:
                    break

            except Exception as e:
                print(f"Error occured {e}")

        return num_steps, rewards
