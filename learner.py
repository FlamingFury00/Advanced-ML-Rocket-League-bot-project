import os
import wandb
import numpy
from typing import Any

import torch
import torch.jit
from torch.nn import Linear, Sequential, ReLU, LeakyReLU

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.default_reward import DefaultReward
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer

from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, VelocityReward, SaveBoostReward, AlignBallGoal
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, LiuDistancePlayerToBallReward, FaceBallReward, TouchBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward, LiuDistanceBallToGoalReward, BallYCoordinateReward
from rlgym.utils.reward_functions import CombinedReward
from rlgym_tools.extra_rewards.jump_touch_reward import JumpTouchReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rewards import TeamSpacingReward
from training_agent import get_agent
from rocket_learn.utils.stat_trackers.common_trackers import AirTouch, AirTouchHeight, BallHeight, BallSpeed, BehindBall, Boost, CarOnGround, Demos, DistToBall, EpisodeLength, GoalSpeed, MaxGoalSpeed, Speed, TimeoutRate, Touch, TouchHeight


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)

config = dict(
    actor_lr=5e-5,
    critic_lr=5e-5,
    n_steps=1_000_000,
    batch_size=20_000,
    minibatch_size=10_000,
    epochs=32,
    gamma=0.9908006132652293,
    clip_range=0.2,
    gae_lambda=0.95,
    vf_coef=1,
    max_grad_norm=0.5
)

if __name__ == "__main__":
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    wandb.login(key="[REDACTED]")
    logger = wandb.init(name="hades-v0", project="ML Rocket League Bot", entity="flamingfury00", id="general", config=config, 
                        settings=wandb.Settings(_disable_stats=True, _disable_meta=True), resume=True, magic=True, force=True, 
                        sync_tensorboard=True, monitor_gym=True)

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    redis = Redis(password="[REDACTED]")
    
    stat_trackers = [AirTouch(), AirTouchHeight(), BallHeight(), BallSpeed(), BehindBall(), Boost(), 
                     CarOnGround(), Demos(), DistToBall(), EpisodeLength(), GoalSpeed(), MaxGoalSpeed(), 
                     Speed(), TimeoutRate(), Touch(), TouchHeight()]

    # ** ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER **
    def obs():
        return ExpandAdvancedObs()

    def rew():
        return CombinedReward(
            (
                VelocityPlayerToBallReward(True),
                VelocityBallToGoalReward(use_scalar_projection=True),
                # LiuDistanceBallToGoalReward(),
                BallYCoordinateReward(),
                VelocityReward(),
                SaveBoostReward(),
                AlignBallGoal(),
                # LiuDistancePlayerToBallReward(),
                FaceBallReward(),
                TouchBallReward(10.0),
                JumpTouchReward(),
                KickoffReward(),
                TeamSpacingReward(),
                EventReward(
                    goal=10.0 * 1,
                    team_goal=100.0,
                    concede=-100.0 + (10.0 * 1),
                    touch=20.0,
                    shot=5.0,
                    save=50.0,
                    demo=10.0,
                    boost_pickup=30.0
                ),
            ),
            (0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

    def act():
        return DiscreteAction()


    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN REDIS DATABASE IS BACKED UP TO DISK
    # -model_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator("ML-bot", redis, obs, rew, act,
                                        logger=logger,
                                        save_every=100,
                                        model_every=100,
                                        clear=False,
                                        max_age=3,
                                        stat_trackers=stat_trackers)

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    split = (3, 3, 3, 3, 3, 2, 2, 2)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 107

    critic = Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, 1)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, total_output),
        SplitLayer(splits=split)
    ), split)

    # CREATE THE OPTIMIZER
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 5e-5},
        {"params": critic.parameters(), "lr": 5e-5}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    # INSTANTIATE THE PPO TRAINING ALGORITHM
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=1_000_000,
        batch_size=20_000,
        minibatch_size=10_000,
        epochs=32,
        gamma=0.9908006132652293,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=1,
        max_grad_norm=0.5,
        logger=logger,
        device="cpu",
    )

    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    
    alg.load("save/ML Rocket League Bot_latest/ML Rocket League Bot_-1/checkpoint.pt")
    alg.agent.optimizer.param_groups[0]["lr"] = 1e-4
    alg.agent.optimizer.param_groups[1]["lr"] = 1e-4
    alg.run(iterations_per_save=10, save_dir="save")