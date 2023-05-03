from time import sleep
import numpy as np
import psutil
import win32gui, win32con, win32process, os, signal
from rlgym.utils.common_values import ORANGE_TEAM, BLUE_TEAM, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, BACK_WALL_Y, CAR_MAX_SPEED, BALL_MAX_SPEED
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import ConditionalRewardFunction
from rlgym.utils import RewardFunction, math
from rlgym.utils.gamestates import PlayerData, GameState
from typing import List

class TeamSpacingReward(RewardFunction):
    def __init__(self, min_spacing: float = 1000) -> None:
        super().__init__()
        self.min_spacing = min_spacing

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != player.car_id and not player.is_demoed and not p.is_demoed:
                separation = np.linalg.norm(player.car_data.position - p.car_data.position)
                if separation < self.min_spacing:
                    reward -= 1-(separation / self.min_spacing)
        return reward