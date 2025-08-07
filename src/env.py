# src/env.py
import numpy as np
import gym
from gym import spaces

class ServerlessTuningEnv(gym.Env):
    def __init__(self, forecast_sequence, latency_sequence, cost_sequence):
        super(ServerlessTuningEnv, self).__init__()

        self.forecast_sequence = forecast_sequence
        self.latency_sequence = latency_sequence
        self.cost_sequence = cost_sequence
        self.max_forecast = max(forecast_sequence)
        self.max_latency = max(latency_sequence)
        self.max_cost = max(cost_sequence)

        self.current_step = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(18)  # 3 mem x 3 timeout x 2 concurrency

        self.reward_mode = "default"  # other options: 'cost_priority', etc.

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        forecast = self.forecast_sequence[self.current_step] / self.max_forecast
        latency = self.latency_sequence[self.current_step] / self.max_latency
        cost = self.cost_sequence[self.current_step] / self.max_cost
        return np.array([forecast, latency, cost], dtype=np.float32)

    def step(self, action):
        forecast = self.forecast_sequence[self.current_step]
        latency = self.latency_sequence[self.current_step]
        cost = self.cost_sequence[self.current_step]

        reward = self._calculate_reward(latency, cost)
        self.current_step += 1
        done = self.current_step >= len(self.forecast_sequence)

        return self._get_obs(), reward, done, False, {}

    def _calculate_reward(self, latency, cost):
        if self.reward_mode == "cost_priority":
            return -cost
        elif self.reward_mode == "latency_priority":
            return -latency
        else:  # default reward: combine both
            return -latency - 0.1 * cost
