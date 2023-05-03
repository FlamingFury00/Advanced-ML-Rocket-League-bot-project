from typing import Any
import numpy

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.default_reward import DefaultReward
from rlgym.utils.state_setters.default_state import DefaultState
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from pretrained_agents.necto.necto_v1 import NectoV1
from pretrained_agents.nexto.nexto_v2 import NextoV2

from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym.utils.state_setters import RandomState, DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, VelocityReward, SaveBoostReward, AlignBallGoal
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, LiuDistancePlayerToBallReward, FaceBallReward, TouchBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward, LiuDistanceBallToGoalReward, BallYCoordinateReward
from rlgym.utils.reward_functions import CombinedReward
from rlgym_tools.extra_rewards.jump_touch_reward import JumpTouchReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rewards import TeamSpacingReward


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    """

    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the 
    learner, and receives updated model parameters when available

    """

    # OPTIONAL ADDITION:
    # LIMIT TORCH THREADS TO 1 ON THE WORKERS TO LIMIT TOTAL RESOURCE USAGE
    # TRY WITH AND WITHOUT FOR YOUR SPECIFIC HARDWARE
    import torch

    torch.set_num_threads(1)
    team_size = 1

    # BUILD THE ROCKET LEAGUE MATCH THAT WILL USED FOR TRAINING
    # -ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    match = Match(
        # game_speed=1,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=WeightedSampleSetter(
                [RandomState(True, True, False), 
                 DefaultState(), 
                 KickoffLikeSetter(), 
                 GoaliePracticeState(False, True, True, False), 
                 HoopsLikeSetter(), 
                 WallPracticeState()], 
                [1.0, 0.5, 0.2, 1.0, 1.0, 1.0]),
        obs_builder=ExpandAdvancedObs(),
        action_parser=DiscreteAction(),
        terminal_conditions=[NoTouchTimeoutCondition(round(2000)),
                             GoalScoredCondition()],
        reward_function=CombinedReward(
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
                    goal=10.0 * team_size,
                    team_goal=100.0,
                    concede=-100.0 + (10.0 * team_size),
                    touch=20.0,
                    shot=5.0,
                    save=50.0,
                    demo=10.0,
                    boost_pickup=30.0
                ),
            ),
            (0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    )
    
    # model_name = "necto-model-30Y.pt"
    # nectov1 = NectoV1(model_string=model_name, n_players=team_size*2)
    
    # model_name1 = "nexto-model.pt"
    # nextov2 = NextoV2(model_string=model_name1, n_players=team_size*2)
    
    # # EACH AGENT AND THEIR PROBABILITY OF OCCURRENCE
    # pretrained_agents = {nectov1: .1, nextov2: .01}

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", password="[REDACTED]")

    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(r, "ML-bot", match,
                       past_version_prob=.2,
                       evaluation_prob=0.01,
                       sigma_target=2,
                    #    pretrained_agents=pretrained_agents,
                       dynamic_gm=False,
                       send_obs=True,
                       streamer_mode=False,
                       deterministic_streamer=False,
                       send_gamestates=False,
                       force_paging=False,
                       auto_minimize=True,
                       local_cache_name="model_database").run()