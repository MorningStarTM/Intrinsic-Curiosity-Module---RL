from ICM.actor_critic import ActorCritic
from ICM.converter import ActionConverter
from ICM.trainer import  Trainer, ICMTrainer
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import warnings
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import PerfectForecastHandler, CSVHandler
from ICM.custom_reward import LossReward, MarginReward


env = grid2op.make("l2rpn_case14_sandbox", 
                reward_class=L2RPNSandBoxScore,
                backend=LightSimBackend(),
                other_rewards={"loss": LossReward, "margin": MarginReward}
                   )
converter = ActionConverter(env=env)

config = {
    "input_dim":env.observation_space.shape.sum(),
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
    'batch_size':256
}

agent = ActorCritic(config=config)
trainer = ICMTrainer(agent=agent, env=env, converter=converter, config=config)
#trainer = Trainer(agent=agent, env=env, converter=converter, config=config)

trainer.train()