import grid2op
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend
from grid2op import Environment
from ICM.converter import ActionConverter
from ICM.actor_critic import ActorCritic, ActorCriticGAT
from ICM.Utils.utils import save_episode_rewards
from ICM.Utils.graph_builder import build_homogeneous_grid_graph, obs_to_pyg_data
from ICM.icm import ICM
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


class Trainer:
    def __init__(self, agent:ActorCritic, env:Environment, converter:ActionConverter, config, use_agent=None):
        self.agent = agent
        self.env = env
        self.config = config
        self.converter = converter
        self.optimizer = self.agent.optimizer#optim.Adam(self.agent.parameters(), lr=self.config['lr'], betas=self.config['betas'])
        self.best_survival_step = 0
        self.update_freq = self.config["update_freq"]#.get("update_freq", 512)
        self.step_counter = 0
        self.episode_rewards = []

        if use_agent is not None:
            self.agent.load_model(use_agent)

    def train(self):
        running_reward = 0
        for i_episode in range(0, self.config['episodes']):
            logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.config['max_ep_len']):
                action = self.agent(obs.to_vect())
                obs_, reward, done, _ = self.env.step(self.converter.act(action))
                self.agent.rewards.append(reward)
                episode_total_reward += reward
                obs = obs_

                if done:
                    break

            logger.info(f"Episode {i_episode} reward: {episode_total_reward}")  
            self.episode_rewards.append(episode_total_reward)  
            # Updating the policy :
            self.optimizer.zero_grad()
            loss = self.agent.calculateLoss(self.config['gamma'])
            loss.backward()
            self.optimizer.step()        
            self.agent.clearMemory()

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(filename="final_actor_critic.pt")    
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0

        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="actor_critic_reward.npy")
        logger.info(f"reward saved at ICM\\episode_reward")



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
    def __init__(self, agent:ActorCritic, icm:ICM, env:Environment, converter:ActionConverter, config) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.converter = converter
        self.icm = icm
        self.actor_optimizer = self.agent.optimizer#optim.Adam(self.agent.parameters(), lr=self.config['lr'], betas=self.config['betas'])
        self.icm_optimizer = self.icm.optimizer
        self.best_survival_step = 0
        self.episode_rewards = []


    def fit(self):
        logger.info("""======================================================= \n
                                    Fit function Invoke \n
                       =======================================================""")
        running_reward = 0
        for i_episode in range(0, self.config['episodes']):
            #logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.config['max_ep_len']):
                action = self.agent(obs.to_vect())
                obs_, reward, done, _ = self.env.step(self.converter.act(action))
                state_, pred_next_state, action_pred, action_ = self.icm(action, obs, obs_)

                intrinsic_reward = self.icm.calc_loss(state_=state_, pred_state=pred_next_state)

                self.icm.memory.remember(state_=state_, pred_state=pred_next_state, actions=action, pred_actions=action_)

                total_reward = reward + intrinsic_reward * self.config['intrinsic_reward_weight']

                self.agent.rewards.append(total_reward)
                episode_total_reward += total_reward
                obs = obs_

                if done:
                    break

            #logger.info(f"Episode {i_episode} reward: {episode_total_reward}")  
            self.episode_rewards.append(episode_total_reward)  
            # Updating the policy :
            self.actor_optimizer.zero_grad()
            self.icm_optimizer.zero_grad()

            icm_loss = self.icm.learn()
            policy_loss = self.agent.calculateLoss(self.config['gamma'])
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
                self.agent.save_checkpoint(filename="final_actor_critic.pt")    
                self.icm.save_checkpoint(filename="final_icm.pt")
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0

        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="final_actor_critic_reward.npy")
        logger.info(f"reward saved at ICM\\episode_reward")




    def train(self, start=0, end=10):
        num_episodes = len(self.env.chronics_handler.subpaths)
        train_step = 0
        for episode_id in range(start, end):

            print(f"Episode ID : {episode_id}")
            self.env.set_id(episode_id)
            obs = self.env.reset()
            done = False

            for i in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode_id}", leave=True):
                train_step += 1
                try:
                    action = self.agent(obs.to_vect()) 
                    obs_, env_reward, done, _ = self.env.step(self.converter.act(action))
                    state_, pred_next_state, action_pred, action_ = self.icm(action, obs, obs_)

                    intrinsic_reward = self.icm.calc_loss(state_=state_, pred_state=pred_next_state)

                    self.icm.memory.remember(state_=state_, pred_state=pred_next_state, actions=action, pred_actions=action_)

                    total_reward = env_reward + intrinsic_reward * 0.001
                    self.agent.rewards.append(total_reward)

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

                    
                    if train_step == 1024:
                        logger.info(f"\n\n###########################################\n Updating at {i}.....\n\n#####################################################")
                        self.actor_optimizer.zero_grad()
                        self.icm_optimizer.zero_grad()

                        icm_loss = self.icm.learn()
                        policy_loss = self.agent.calculateLoss(self.config['gamma'])
                        total_loss = icm_loss + policy_loss

                        total_loss.backward()
                        self.actor_optimizer.step()
                        self.icm_optimizer.step()

                        self.agent.clearMemory()
                        self.icm.memory.clear_memory()
                        train_step = 0



                except NoForecastAvailable as e:
                    logger.info(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue

                except Grid2OpException as e:
                    logger.info(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue 


            if episode_id!=0 and episode_id % 5 == 0:
                print(f"\n\n#############################################\n Saving the Agent \n\n#############################################\n\n")
                self.agent.save_checkpoint(filename=f"icm_actor_critic_{episode_id}.pt")
                self.icm.save_checkpoint(filename=f"icm_{episode_id}.pt")
                




class GraphAgentTrainer:
    def __init__(self, agent:ActorCriticGAT, env:Environment, converter:ActionConverter, config) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.converter = converter
        self.optimizer = self.agent.optimizer
        self.best_survival_step = 0
        self.episode_rewards = []
        self.episode_lenths = []
        self.episode_reasons  = []   
        self.episode_path = self.config['episode_path']
        os.makedirs(self.episode_path, exist_ok=True)
        logger.info(f"Episode path : {self.episode_path}")  
    
    
    def train(self):
        running_reward = 0
        update_every = self.config["update_freq"]
        for i_episode in range(0, self.config['episodes']):
            logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0
            
            
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            

            for t in range(self.config['max_ep_len']):
                data = build_homogeneous_grid_graph(obs, self.env, device=self.agent.device, danger_thresh=0.98)
                if data.num_nodes == 0:
                    # minimal 1-node fallback (keeps training alive)
                    data.x = torch.zeros(1, self.agent.config['input_dim'], device=self.agent.device)
                    data.edge_index = torch.empty(2, 0, dtype=torch.long, device=self.agent.device)

                batch = getattr(data, "batch",
                torch.zeros(data.num_nodes, dtype=torch.long, device=self.agent.device))

                with torch.cuda.amp.autocast(enabled=True):
                    action = self.agent(data.x, data.edge_index, batch)
                obs_, reward, done, info = self.env.step(self.converter.act(action))
                self.agent.rewards.append(reward)
                episode_total_reward += reward
                obs = obs_
                

                if done:
                    break

            logger.info(f"Episode {i_episode} reward: {episode_total_reward}")  
            self.episode_rewards.append(episode_total_reward) 
            # tag why the episode ended (best-effort; keys depend on env)
            reason = "done" if done else "max_ep_len"
            # If the env provides richer signals, prefer them:
            if isinstance(info, dict):
                if info.get("is_illegal", False):
                    reason = "illegal"
                elif info.get("is_ambiguous", False):
                    reason = "ambiguous"
                elif info.get("is_blackout", False):
                    reason = "blackout"
                elif info.get("is_game_over", False):
                    reason = "game_over"
                elif info.get("is_last", False) or info.get("is_final_observation", False):
                    reason = "end_of_chronic"
            
            self.episode_lenths.append(t+1)   
            self.episode_reasons.append(reason)          

            # Updating the policy :
            if (t+1) % update_every == 0 or done:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=True):
                    loss = self.agent.calculateLoss(self.config['gamma'])
                loss.backward()
                #scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                self.optimizer.step()        
                #scaler.step(self.optimizer)
                #scaler.update()
                self.agent.clearMemory()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(filename="gat_actor_critic.pt")    
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0

        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="actor_critic_gat_reward.npy")
        np.save(os.path.join(self.episode_path, "actor_critic_gat_lengths.npy"), np.array(self.episode_lenths, dtype=np.int32))

        df = pd.DataFrame({
            "episode": list(range(len(self.episode_rewards))),
            "reward": self.episode_rewards,
            "length": self.episode_lenths,
            "reason": self.episode_reasons
        })

        csv_path = os.path.join(self.episode_path, "actor_critic_gat_stats.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"episode stats saved at {self.episode_path}")


        logger.info(f"reward saved at ICM\\episode_reward")
        logger.info(f"Saved training stats to {csv_path}")
    