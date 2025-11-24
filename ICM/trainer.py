import grid2op
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend
from grid2op import Environment
from ICM.converter import ActionConverter
from ICM.actor_critic import ActorCritic, ActorCriticGAT, ActorCriticUP
from ICM.Utils.utils import save_episode_rewards
from ICM.Utils.graph_builder import build_homogeneous_grid_graph, obs_to_pyg_data
from ICM.icm import ICM, GraphACICM
import torch.optim as optim
from grid2op.Exceptions import *
from tqdm import tqdm
from ICM.Utils.logger import logger
import random
import inspect
import numpy as np
import pandas as pd
import torch
import os
import re
import torch.nn.functional as F


class Trainer:
    def __init__(self, agent:ActorCritic, env:Environment, converter:ActionConverter, config, use_agent=None):
        self.agent = agent
        self.env = env
        self.config = config
        self.converter = converter
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.config['lr'], betas=self.config['betas'])
        self.best_survival_step = 0
        self.update_freq = self.config["update_freq"]#.get("update_freq", 512)
        self.step_counter = 0
        self.episode_rewards = []
        self.episode_steps = []  

        if use_agent is not None:
            self.agent.load_model(use_agent)

    def train(self):
        running_reward = 0
        actions = []
        for i_episode in range(0, self.config['episodes']):
            logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.config['max_ep_len']):
                action = self.agent(obs.to_vect())
                actions.append(action)
                obs_, reward, done, _ = self.env.step(self.converter.act(action))
                self.agent.rewards.append(reward)
                episode_total_reward += reward
                obs = obs_

                if done:
                    np.save("ICM\\episode_reward\\actions.npy",
                            np.array([int(a) for a in actions], dtype=np.int32))
                    logger.info(f"Saved actions for episode {i_episode} at ICM\\episode_reward\\actions.npy")
                    break

            logger.info(f"Episode {i_episode} reward: {episode_total_reward}")  
            self.episode_rewards.append(episode_total_reward)  
            # Updating the policy :
            self.optimizer.zero_grad()
            loss = self.agent.calculateLoss(self.config['gamma'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optimizer.step()        
            self.agent.clearMemory()

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(optimizer=self.optimizer, filename="final_actor_critic.pt")    
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0
            
            survival_steps = t + 1          # because t is 0-indexed
            self.episode_steps.append(survival_steps)

        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="actor_critic_reward.npy")
        logger.info(f"reward saved at ICM\\episode_reward")
        os.makedirs("ICM\\episode_reward", exist_ok=True)
        np.save("ICM\\episode_reward\\actor_critic_steps.npy", np.array(self.episode_steps, dtype=int))




    def train_ep(self, s_epi, t_epi):
        num_episodes = len(self.env.chronics_handler.subpaths)

        for episode_id in range(s_epi, t_epi):

            logger.info(f"Episode ID : {episode_id}")
            self.env.set_id(episode_id)
            obs = self.env.reset()
            done = False
            episode_total_reward = 0


            for i in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode_id}", leave=True):
                
                try:
                    action = self.agent(obs.to_vect()) 
                    obs_, reward, done, _ = self.env.step(self.converter.act(action))
                    self.agent.rewards.append(reward)
                    episode_total_reward += reward
                    self.step_counter += 1
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

                    if self.step_counter % self.update_freq == 0:
                        logger.info(f"\###########################################\n updating at {i}.....")
                        self.optimizer.zero_grad()
                        loss = self.agent.calculateLoss()
                        logger.info(f"Loss calculated : {loss}")
                        loss.backward()
                        self.optimizer.step()
                        self.agent.clearMemory()

                except NoForecastAvailable as e:
                    logger.error(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue

                except Grid2OpException as e:
                    logger.error(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue 


            if episode_id!=0 and episode_id % 5 == 0:
                logger.info(f"\n\n#############################################\n\n Evaluating the Agent\n\n#############################################\n\n")
                num_steps, rewards = self.evaluate()
                logger.info(f"\n################################\n Number of steps agent survived is {num_steps}")
                if self.best_survival_step < num_steps:
                    logger.info(f"Agent survived {num_steps}/{self.env.max_episode_duration()} steps")
                    self.agent.save_model(model_name=f"actor_critic_{episode_id}_{num_steps}")
                    self.best_survival_step = num_steps

            self.episode_rewards.append(episode_total_reward)
            logger.info(f"episode reward stored")
        self.agent.save_model(f"actor_critic_episode_{t_epi}")
        logger.info(f"last episode agent saved at {t_epi}")
        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="episode_rewards_361_421.npy")

            



    def evaluate(self):
        num_steps = 0
        rewards = []
        paths = self.env.chronics_handler.subpaths
        test_path = random.choice(paths[900:])
        logger.info(f"Selected Chronics : {test_path}")

        try:
            self.env.set_id(test_path)
            logger.info(f"Selected Chronic loaded")
        except Exception as e:
            logger.error("Error occured {e}")
            logger(f"{self.__class__.__name__}.{__name__} Error Occured")

        obs = self.env.reset()
        reward = self.env.reward_range[0]
        done = False

        for i in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {test_path}", leave=True):
            num_steps += 1
            try:
                action = self.agent(obs.to_vect()) 
                obs_, reward, done, _ = self.env.step(self.converter.act(action))
                rewards.append(reward)
                obs = obs_

                if done:
                    break

            except Exception as e:
                logger.error(f"Error occured {e}")

        return num_steps, rewards



class ICMTrainer:
    def __init__(self, env:Environment, converter:ActionConverter, agent_type:str, model_config, icm_config) -> None:
        self.env = env
        self.model_config = model_config
        self.icm_config = icm_config
        self.converter = converter

        if agent_type == 'GAT':

            self.agent = ActorCriticGAT(config=model_config)
            self.actor_optimizer = self.agent.optimizer

            self.icm = GraphACICM(config=icm_config)
            self.icm_optimizer = self.icm.optimizer
            logger.info("Graph Actor-Critic with graph ICM Trainer Initialized")

        else:
            self.agent = ActorCriticUP()
            self.actor_optimizer = optim.Adam(self.agent.parameters(), lr=self.model_config['lr'], betas=self.model_config['betas'])
            self.icm = ICM(config=self.icm_config)
            self.icm_optimizer = self.icm.optimizer
            logger.info("Actor-Critic with ICM Trainer Initialized")
        
        self.best_survival_step = 0
        self.episode_rewards = []
        self.episode_lenths = []
        self.episode_reasons = []
        self.episode_path = self.model_config['episode_path']
        os.makedirs(self.episode_path, exist_ok=True)
        logger.info(f"Episode path : {self.episode_path}")


    def fit(self, start=0, end=100):
        logger.info("""======================================================= \n
                                    Fit function Invoke \n
                       =======================================================""")
        regex = self.make_chronic_regex(start=start, end=end)
        self.env.chronics_handler.set_filter(lambda p, regex=regex: re.match(regex, p) is not None)
        self.env.chronics_handler.reset()

        running_reward = 0
        for i_episode in range(0, self.model_config['episodes']):
            #logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.model_config['max_ep_len']):
                action = self.agent(obs.to_vect())
                obs_, reward, done, _ = self.env.step(self.converter.act(action))
                state_, pred_next_state, action_pred, action_ = self.icm(action, obs, obs_)

                intrinsic_reward = self.icm.calc_loss(state_=state_, pred_state=pred_next_state)

                self.icm.memory.remember(state_=state_, pred_state=pred_next_state, actions=action, pred_actions=action_)

                total_reward = reward + intrinsic_reward * self.icm_config['intrinsic_reward_weight']

                self.agent.rewards.append(total_reward)
                episode_total_reward += total_reward
                obs = obs_

                if done:
                    break

            #logger.info(f"Episode {i_episode} reward: {episode_total_reward}")  
            self.episode_rewards.append(episode_total_reward)  
            self.episode_lenths.append(t + 1)
            # Updating the policy :
            self.actor_optimizer.zero_grad()
            self.icm_optimizer.zero_grad()

            icm_loss = self.icm.learn()
            policy_loss = self.agent.calculateLoss(self.model_config['gamma'])
            total_loss = policy_loss + icm_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 1.0)
            self.actor_optimizer.step()     
            self.icm_optimizer.step()

            self.agent.clearMemory()
            self.icm.memory.clear_memory()

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(optimizer=self.actor_optimizer, filename="final_actor_critic.pt")    
                self.icm.save_checkpoint(filename="final_icm.pt")
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0

        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="final_actor_critic_reward.npy")
        np.save(os.path.join(self.episode_path, "final_actor_critic_lengths.npy"),
                np.array(self.episode_lenths, dtype=np.int32)) 
        logger.info(f"reward saved at ICM\\episode_reward")


    def make_chronic_regex(self, start, end):
        ids = "|".join(f"{i:04d}" for i in range(start, end+1))
        return rf".*({ids}).*"


    

    


