import math
import numpy as np
from osim.env import ProstheticsEnv
from gym.spaces import Box

from envs.prosthetics_preprocess import preprocess_obs, \
    euler_angles_to_rotation_matrix, get_simbody_state


class ProstheticsEnvWrap:
    def __init__(
            self,
            frame_skip=1,
            visualize=False,
            randomized_start=False,
            max_episode_length=300,
            reward_scale=0.1,
            death_penalty=0.0,
            living_bonus=0.0,
            side_deviation_penalty=0.0,
            crossing_legs_penalty=0.0,
            bending_knees_bonus=0.0,
            side_step_penalty=False,
            action_fn=None,
            observe_time=False,
            model="3D"):

        self.model = model
        self.visualize = visualize
        self.randomized_start = randomized_start
        self.env = ProstheticsEnv(visualize=visualize, integrator_accuracy=1e-3)
        self.env.change_model(
            model=self.model, prosthetic=True, difficulty=0,
            seed=np.random.randint(200))

        self.frame_skip = frame_skip
        self.observe_time = observe_time
        self.max_ep_length = max_episode_length - 2

        self.observation_space = Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            shape=(341 + int(observe_time),))
        self.action_space = Box(
            low=self.env.action_space.low[0],
            high=self.env.action_space.high[0],
            shape=(19,))

        # reward shaping
        self.reward_scale = reward_scale
        self.death_penalty = np.abs(death_penalty)
        self.living_bonus = living_bonus
        self.side_dev_coef = side_deviation_penalty
        self.cross_legs_coef = crossing_legs_penalty
        self.bending_knees_coef = bending_knees_bonus
        self.side_step_penalty = side_step_penalty

        self.episodes = 1
        self.ep2reload = 10

        # ddpg different output activations support
        if action_fn == "tanh":
            action_mean = .5
            action_std = .5
            self.action_handler = lambda x: x * action_std + action_mean
        else:
            self.action_handler = lambda x: x

    def reset(self):
        self.time_step = 0

        if self.episodes % self.ep2reload == 0:
            self.env = ProstheticsEnv(
                visualize=self.visualize, integrator_accuracy=1e-3)
            self.env.change_model(
                model=self.model, prosthetic=True, difficulty=0,
                seed=np.random.randint(200))

        state_desc = self.env.reset(project=False)
        if self.randomized_start:
            state = get_simbody_state(state_desc)
            noise = np.random.normal(scale=0.1, size=72)
            noise[3:6] = 0
            state = (np.array(state) + noise).tolist()
            simbody_state = self.env.osim_model.get_state()
            obj = simbody_state.getY()
            for i in range(72):
                obj[i] = state[i]
            self.env.osim_model.set_state(simbody_state)

        observation = preprocess_obs(state_desc)
        if self.observe_time:
            observation.append(-1.0)

        return observation

    def step(self, action):
        reward = 0
        reward_origin = 0

        action = self.action_handler(action)
        action = np.clip(action, 0.0, 1.0)

        for i in range(self.frame_skip):
            observation, r, _, info = self.env.step(action, project=False)
            reward_origin += r
            done = self.is_done(observation)
            reward += self.shape_reward(r)
            if done:
                self.episodes = self.episodes + 1
                break

        observation = preprocess_obs(observation)
        reward *= self.reward_scale
        info["reward_origin"] = reward_origin
        self.time_step += 1

        if self.observe_time:
            observation.append(
                (self.time_step / self.max_ep_length - 0.5) * 2.)

        return observation, reward, done, info

    def is_done(self, observation):
        pelvis_y = observation["body_pos"]["pelvis"][1]
        if self.time_step * self.frame_skip > self.max_ep_length:
            return True
        elif pelvis_y < 0.6:
            return True
        return False

    def shape_reward(self, reward):
        state_desc = self.env.get_state_desc()

        # death penalty
        if self.time_step * self.frame_skip < self.max_ep_length:
            reward -= self.death_penalty
        else:
            reward += self.living_bonus

        # deviation from forward direction penalty
        vy, vz = state_desc['body_vel']['pelvis'][1:]
        side_dev_penalty = (vy ** 2 + vz ** 2)
        reward -= self.side_dev_coef * side_dev_penalty

        # crossing legs penalty
        pelvis_xy = np.array(state_desc['body_pos']['pelvis'])
        left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xy
        right = np.array(state_desc['body_pos']['pros_foot_r']) - pelvis_xy
        axis = np.array(state_desc['body_pos']['head']) - pelvis_xy
        cross_legs_penalty = np.cross(left, right).dot(axis)
        if cross_legs_penalty > 0:
            cross_legs_penalty = 0.0
        reward += self.cross_legs_coef * cross_legs_penalty

        # bending knees bonus
        r_knee_flexion = np.minimum(state_desc['joint_pos']['knee_r'][0], 0.)
        l_knee_flexion = np.minimum(state_desc['joint_pos']['knee_l'][0], 0.)
        bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        reward += self.bending_knees_coef * bend_knees_bonus

        # side step penalty
        if self.side_step_penalty:
            rx, ry, rz = state_desc['body_pos_rot']['pelvis']
            R = euler_angles_to_rotation_matrix([rx, ry, rz])
            reward *= (1.0 - math.fabs(R[2, 0]))

        return reward


ENV = ProstheticsEnvWrap
