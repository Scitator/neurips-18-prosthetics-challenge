import math
import random
import numpy as np
from osim.env import ProstheticsEnv
from gym.spaces import Box

from catalyst.utils.misc import set_global_seeds
from catalyst.contrib.registry import Registry
from envs.prosthetics_preprocess import preprocess_obs_round2, \
    euler_angles_to_rotation_matrix, get_simbody_state


SEED_RANGE = 2 ** 32 - 2


@Registry.environment
class ProstheticsEnvWrap:
    def __init__(
            self,
            frame_skip=1,
            visualize=False,
            randomized_start=False,
            max_episode_length=1000,
            reward_scale=0.1,
            death_penalty=0.0,
            living_bonus=0.0,
            crossing_legs_penalty=0.0,
            bending_knees_bonus=0.0,
            left_knee_bonus=0.,
            right_knee_bonus=0.,
            bonus_for_knee_angles_scale=0.,
            bonus_for_knee_angles_angle=0.,
            activations_penalty=0.,
            max_reward=10.0,
            action_fn=None,
            observe_time=False,
            model="3D"):

        self.model = model
        self.visualize = visualize
        self.randomized_start = randomized_start
        self.env = ProstheticsEnv(visualize=visualize, integrator_accuracy=1e-3)
        seed = random.randrange(SEED_RANGE)
        set_global_seeds(seed)
        self.env.change_model(
            model=self.model, prosthetic=True, difficulty=1,
            seed=seed)

        self.frame_skip = frame_skip
        self.observe_time = observe_time
        hotfix_flag = 1 - frame_skip % 2
        self.max_ep_length = max_episode_length - 2 - hotfix_flag * 2

        self.observation_space = Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            shape=(343 + int(observe_time),))
        self.action_space = Box(
            low=self.env.action_space.low[0],
            high=self.env.action_space.high[0],
            shape=(19,))

        # reward shaping
        self.reward_scale = reward_scale
        self.death_penalty = np.abs(death_penalty)
        self.living_bonus = living_bonus
        self.cross_legs_coef = crossing_legs_penalty
        self.bending_knees_coef = bending_knees_bonus
        self.max_reward = max_reward
        self.activations_penalty = activations_penalty
        self.left_knee_bonus = left_knee_bonus
        self.right_knee_bonus = right_knee_bonus
        self.bonus_for_knee_angles_scale = bonus_for_knee_angles_scale
        self.knees_angle_bonus = bonus_for_knee_angles_angle

        self.episodes = 1
        self.ep2reload = 5

        # ddpg different output activations support
        action_fn = action_fn.lower()
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
            seed = random.randrange(SEED_RANGE)
            set_global_seeds(seed)
            self.env.change_model(
                model=self.model, prosthetic=True, difficulty=1,
                seed=seed)

        state_desc = self.env.reset(project=False)
        if self.randomized_start:
            state = get_simbody_state(state_desc)

            amplitude = random.gauss(0.8, 0.05)
            direction = random.choice([-1., 1])
            amplitude_knee = random.gauss(-1.2, 0.05)
            state[4] = 0.8
            state[6] = amplitude * direction  # right leg
            state[9] = amplitude * direction * (-1.)  # left leg
            state[13] = amplitude_knee if direction == 1. else 0  # right knee
            state[14] = amplitude_knee if direction == -1. else 0  # left knee

            # noise = np.random.normal(scale=0.1, size=72)
            # noise[3:6] = 0
            # noise[6] = np.random.uniform(-1., 1., size=1)
            # noise[9] = np.random.uniform(-1., 1., size=1)
            # noise[13] = -np.random.uniform(0., 1., size=1)  # knee_r
            # noise[14] = -np.random.uniform(0., 1., size=1)  # knee_l
            # state = (np.array(state) + noise).tolist()

            simbody_state = self.env.osim_model.get_state()
            obj = simbody_state.getY()
            for i in range(72):
                obj[i] = state[i]
            self.env.osim_model.set_state(simbody_state)

        observation = preprocess_obs_round2(state_desc)
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
            reward += self.shape_reward(r, done)
            if done:
                self.episodes = self.episodes + 1
                break

        observation = preprocess_obs_round2(observation)
        reward *= self.reward_scale
        info["reward_origin"] = reward_origin
        self.time_step += 1

        if self.observe_time:
            time_pass = (self.time_step * self.frame_skip) / self.max_ep_length
            observation.append(
                (time_pass - 0.5) * 2.)

        return observation, reward, done, info

    def is_done(self, observation):
        pelvis_y = observation["body_pos"]["pelvis"][1]
        if self.time_step * self.frame_skip > self.max_ep_length:
            return True
        elif pelvis_y < 0.6:
            return True
        return False

    def shape_reward(self, reward, done):
        state_desc = self.env.get_state_desc()

        # death penalty
        if done and self.time_step * self.frame_skip < self.max_ep_length:
            reward -= self.death_penalty
        else:
            reward += self.living_bonus

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
        r_knee_flexion = np.minimum(state_desc['joint_pos']['knee_r'][0],
                                    0.)
        l_knee_flexion = np.minimum(state_desc['joint_pos']['knee_l'][0],
                                    0.)
        bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        reward += self.bending_knees_coef * bend_knees_bonus

        reward += \
            self.bonus_for_knee_angles_scale \
            * math.exp(-((r_knee_flexion + self.knees_angle_bonus) * 6.0) ** 2)
        reward += \
            self.bonus_for_knee_angles_scale \
            * math.exp(-((l_knee_flexion + self.knees_angle_bonus) * 6.0) ** 2)

        r_knee_flexion = math.fabs(state_desc['joint_vel']['knee_r'][0])
        l_knee_flexion = math.fabs(state_desc['joint_vel']['knee_l'][0])
        reward += r_knee_flexion * self.right_knee_bonus
        reward += l_knee_flexion * self.left_knee_bonus

        reward -= np.sum(
            np.array(self.env.osim_model.get_activations()) ** 2
        ) * self.activations_penalty

        reward = reward - 10.0 + self.max_reward

        return reward


ENV = ProstheticsEnvWrap
