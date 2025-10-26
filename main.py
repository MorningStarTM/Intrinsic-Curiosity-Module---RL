from ICM.actor_critic import ActorCritic, ActorCriticGAT
from ICM.converter import ActionConverter
from ICM.icm import ICM 
from ICM.trainer import  Trainer, ICMTrainer, GraphAgentTrainer
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import warnings
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import PerfectForecastHandler, CSVHandler
from ICM.custom_reward import LossReward, MarginReward
import argparse


env = grid2op.make("l2rpn_case14_sandbox", 
                #reward_class=L2RPNSandBoxScore,
                backend=LightSimBackend(),
                #other_rewards={"loss": LossReward, "margin": MarginReward}
                   )
converter = ActionConverter(env=env)

config = {
    "input_dim":493, #env.observation_space.shape.sum(),
    "action_dim":converter.n,
    "gamma": 0.99,
    "lr": 0.0003,
    "betas": (0.9, 0.999),
    "update_freq": 512,
    "save_path":"ICM\models",
    'episodes': 10000,
    'max_ep_len':10000,
    'icm_lr':1e-4,
    'beta':1e-4,
    'alpha':1e-4,
    'batch_size':256,
    'intrinsic_reward_weight':1,
}


graph_config = {
    "input_dim":11,
    "action_dim":converter.n,
    "gamma": 0.99,
    "lr": 0.0003,
    "betas": (0.9, 0.999),
    "update_freq": 512,
    "save_path":"ICM\models",
    "episode_path":"ICM\episode_length",
    'episodes': 10000,
    'max_ep_len':10000,
    'icm_lr':1e-4,
    'beta':1e-4,
    'alpha':1e-4,
    'batch_size':256,
    'intrinsic_reward_weight':1,
}


def actor_critic_train():
    agent = ActorCritic(config=config)
    trainer = Trainer(agent=agent, env=env, converter=converter, config=config)
    trainer.train()

def graph_actor_critic_train():
    agent = ActorCriticGAT(config=graph_config)
    trainer = GraphAgentTrainer(agent=agent, env=env, converter=converter, config=graph_config)
    trainer.train()


def icm_actor_critic_train(icm_file="icm_230.pt", ac_file="icm_actor_critic_230.pt", type='fit', start=None, end=None):
    agent = ActorCritic(config=config)
    agent.load_checkpoint(folder_name="ICM\\models", filename=ac_file)
    icm = ICM(config=config)
    icm.load_checkpoint(folder_name="ICM\\models", filename=icm_file)
    trainer = ICMTrainer(agent=agent, icm=icm, env=env, converter=converter, config=config)
    if type == 'fit':
        trainer.fit()
    else:
        trainer.train(start=start, end=end)

def icm_actor_critic_train_from_scratch(type='fit', start=None, end=None):
    agent = ActorCritic(config=config)
    
    icm = ICM(config=config)
    
    trainer = ICMTrainer(agent=agent, icm=icm, env=env, converter=converter, config=config)
    if type == 'fit':
        trainer.fit()
    else:
        trainer.train(start=start, end=end)



def main():
    parser = argparse.ArgumentParser(description="Run training")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Actor-Critic
    ac_parser = subparsers.add_parser("actor_critic")

    gac_parser = subparsers.add_parser("gat_actor_critic")
    

    # ICM + Actor-Critic
    icm_parser = subparsers.add_parser("icm")
    icm_parser.add_argument("--icm_file", default="icm_230.pt")
    icm_parser.add_argument("--ac_file", default="icm_actor_critic_230.pt")
    icm_parser.add_argument("--type", choices=["fit", "train"], default="fit")
    icm_parser.add_argument("--start", type=int, default=None)
    icm_parser.add_argument("--end", type=int, default=None)

    icm_ac_parser = subparsers.add_parser("icm_ac_scratch")
    icm_ac_parser.add_argument("--type", choices=["fit", "train"], default="fit")
    icm_ac_parser.add_argument("--start", type=int, default=None)
    icm_ac_parser.add_argument("--end", type=int, default=None)


    args = parser.parse_args()

    if args.mode == "actor_critic":
        actor_critic_train()

    if args.mode == "gat_actor_critic":
        graph_actor_critic_train()


    elif args.mode == "icm":
        icm_actor_critic_train(
            icm_file=args.icm_file,
            ac_file=args.ac_file,
            type=args.type,
            start=args.start,
            end=args.end,
        )
    elif args.mode == "icm_ac_scratch":
        icm_actor_critic_train_from_scratch(
            type=args.type,
            start=args.start,
            end=args.end,
        )

if __name__ == "__main__":
    main()
