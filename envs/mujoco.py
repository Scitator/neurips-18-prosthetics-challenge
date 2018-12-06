#!/usr/bin/env python

import gym
import time


class Mujoco:
    def __init__(
            self,
            env_name="swimmer",
            frame_skip=1,
            visualize=False,
            reward_scale=1,
            step_delay=0.1):

        words = [word.capitalize() for word in env_name.split("_")]
        gym_name = "".join(words) + "-v2"
        self.env = gym.make(gym_name)

        self.visualize = visualize
        self.frame_skip = frame_skip
        self.reward_scale = reward_scale
        self.step_delay = step_delay

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.time_step = 0
        self.total_reward = 0

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        return self.env.reset()

    def step(self, action):
        time.sleep(self.step_delay)
        reward = 0
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action)
            if self.visualize:
                self.env.render()
            reward += r
            if done: break
        self.total_reward += reward
        self.time_step += 1
        info["reward_origin"] = reward
        reward *= self.reward_scale
        return observation, reward, done, info


ENV = Mujoco
