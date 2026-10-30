"""
RecurrentPPO (sb3_contrib) on Safety-Gymnasium Goal tasks, with a wrapper
that restricts the lidar to create partial observability.

Setup (Python 3.10 -- safety-gymnasium fails to import on 3.11+):
    py -3.10 -m venv sg-env
    sg-env\\Scripts\\activate
    python -m pip install --upgrade pip
    python -m pip install safety-gymnasium "numpy<2"
    python -m pip install "gymnasium==0.29.1" stable-baselines3 sb3-contrib tensorboard "numpy<2"
(pip will warn that safety-gymnasium wants gymnasium 0.28.1; 0.29.1 works fine.)

Observation layout for SafetyPointGoal1-v0 (flat Box of 60):
    accelerometer(3) velocimeter(3) gyro(3) magnetometer(3)
    goal_lidar(16) hazards_lidar(16) vases_lidar(16)

Lidar encoding ("pseudo" lidar, the default):
    - 16 bins around the agent. Bin i covers angles [i*22.5, (i+1)*22.5) degrees,
      measured counter-clockwise from the agent's forward (+x) direction.
    - Each bin reads 1 - d / max_dist for the closest object in that direction
      (max_dist = 3), and 0 if nothing is there. Neighboring bins also get a
      partial reading because of aliasing.
"""

import numpy as np
import gymnasium as gym
import safety_gymnasium
from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize,
)
from sb3_contrib import RecurrentPPO

from custom_goal_env import register_goal_env

# ENV_ID = "SafetyPointGoal1-v0"   # Point agent, goal + 8 hazards + 1 vase
# Custom layout (registered at import time so subprocess workers see it too):
pillar_pos=[(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (0.0, 1.5), (-1.5, 0.0), (1.5, 0.0), (0.0, -1.5)]
ENV_ID = register_goal_env("ObstacleGoal1", hazards=4, vases=1, pillars=4, gremlins=1,
                           extents=2.0, lidar_bins=16, lidar_type="pseudo")

# ---- Observability settings: these are your experimental knobs ------------
FULL_OBS = dict(fov_deg=360.0, sense_range=None)
PARTIAL_OBS = dict(
    fov_deg=90.0,          # only see objects within +/-45 degrees of forward
    sense_range=1.0,       # only see objects closer than 1.0 m (max is 3.0)
    noise_std=0.0,         # Gaussian noise on nonzero readings
    dropout_prob=0.0,      # chance each bin reads 0 on a given step
    lidar_keys=None,       # None = all lidars; e.g. ["goal_lidar"] for just the goal
)
OBS_CONFIG = PARTIAL_OBS


class PartialLidar(gym.ObservationWrapper):
    """Restrict field of view, range, and reliability of Safety-Gymnasium lidars.

    Works on the flat observation produced by SafetyGymnasium2Gymnasium by
    locating each '*_lidar' block using the env's observation-space dict.
    """

    def __init__(self, env, space_dict, lidar_max_dist=3.0, fov_deg=360.0,
                 sense_range=None, noise_std=0.0, dropout_prob=0.0,
                 lidar_keys=None, seed=None):
        super().__init__(env)

        # Map each observation component to its slice of the flat vector.
        self.slices, start = {}, 0
        for key, space in space_dict.spaces.items():
            size = int(np.prod(space.shape))
            self.slices[key] = slice(start, start + size)
            start += size
        assert start == env.observation_space.shape[0], "Observation layout mismatch"

        if lidar_keys is None:
            lidar_keys = [k for k in self.slices if k.endswith("_lidar")]
        self.lidar_keys = lidar_keys

        # Field-of-view mask: keep bins whose center lies within +/- fov/2 of forward.
        half_fov = np.deg2rad(fov_deg) / 2.0
        self.fov_masks = {}
        for key in self.lidar_keys:
            n = self.slices[key].stop - self.slices[key].start
            centers = (np.arange(n) + 0.5) * (2 * np.pi / n)
            centers = (centers + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi)
            self.fov_masks[key] = (np.abs(centers) <= half_fov + 1e-9).astype(np.float32)

        # Range limit: reading = 1 - d/max_dist, so d < r  <=>  reading > 1 - r/max_dist.
        if sense_range is None:
            self.min_reading = 0.0
        else:
            self.min_reading = max(0.0, 1.0 - sense_range / lidar_max_dist)

        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.rng = np.random.default_rng(seed)

        # Kept for visualization (see visualize_agent.py).
        self.fov_deg = fov_deg
        self.sense_range = sense_range
        self.lidar_max_dist = lidar_max_dist
        self.last_full_obs = None

        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.astype(np.float32),
            high=env.observation_space.high.astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32, copy=True)
        self.last_full_obs = obs.copy()
        for key in self.lidar_keys:
            s = self.slices[key]
            r = obs[s] * self.fov_masks[key]
            r[r <= self.min_reading] = 0.0
            if self.noise_std > 0:
                noisy = np.clip(r + self.rng.normal(0, self.noise_std, r.shape), 0.0, 1.0)
                r = np.where(r > 0, noisy, 0.0)
            if self.dropout_prob > 0:
                r[self.rng.random(r.shape) < self.dropout_prob] = 0.0
            obs[s] = r
        return obs


def make_env(rank=0, seed=0, obs_config=None, render_mode=None, **make_kwargs):
    """make_kwargs go to safety_gymnasium.make, e.g. width, height, camera_name."""
    def _init():
        safe_env = safety_gymnasium.make(ENV_ID, render_mode=render_mode, **make_kwargs)
        task = safe_env.unwrapped.task
        space_dict = task.obs_info.obs_space_dict
        max_dist = task.lidar_conf.max_dist

        env = SafetyGymnasium2Gymnasium(safe_env)   # 5-tuple step; cost -> info["cost"]
        env = PartialLidar(env, space_dict, lidar_max_dist=max_dist,
                           seed=seed + rank, **(obs_config or {}))
        env.reset(seed=seed + rank)
        return env
    return _init


def inspect_env():
    """Print the observation layout so you can check what each lidar block is."""
    env = make_env(obs_config=OBS_CONFIG)()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    for key, s in env.slices.items():
        print(f"  {key:15s} indices {s.start:3d}-{s.stop - 1:3d}")
    print("Lidar FOV masks (1 = visible bin):")
    for key in env.lidar_keys:
        print(f"  {key:15s} {env.fov_masks[key].astype(int)}")
    env.close()


def visualize_environment(n_steps=1000, seed=0):
    """Open the environment viewer and move the agent with random actions."""
    env = make_env(seed=seed, obs_config=OBS_CONFIG, render_mode="human")()
    try:
        obs, info = env.reset(seed=seed)
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


def train(total_timesteps=1_000_000, n_envs=8, use_subproc=False,
          save_path="rppo_safety_goal"):
    env_fns = [make_env(rank=i, obs_config=OBS_CONFIG) for i in range(n_envs)]
    # SubprocVecEnv is faster but does not work from Jupyter on Windows.
    venv = SubprocVecEnv(env_fns) if use_subproc else DummyVecEnv(env_fns)
    venv = VecMonitor(venv, info_keywords=("cost",))
    # Accelerometer etc. have very different scales from lidar; normalize.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        n_steps=512,
        batch_size=1024,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        verbose=1,
        tensorboard_log="./tb_logs",
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    venv.save(save_path + "_vecnormalize.pkl")
    venv.close()
    return save_path


def evaluate(save_path="rppo_safety_goal", n_episodes=5, render=False):
    render_mode = "human" if render else None
    venv = DummyVecEnv([make_env(seed=1000, obs_config=OBS_CONFIG, render_mode=render_mode)])
    venv = VecNormalize.load(save_path + "_vecnormalize.pkl", venv)
    venv.training = False        # freeze normalization statistics
    venv.norm_reward = False     # report raw rewards

    model = RecurrentPPO.load(save_path, env=venv)

    obs = venv.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    ep_return, ep_cost, results = 0.0, 0.0, []

    while len(results) < n_episodes:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True,
        )
        obs, rewards, dones, infos = venv.step(action)
        ep_return += float(rewards[0])
        ep_cost += float(infos[0].get("cost", 0.0))
        episode_starts = dones
        if dones[0]:
            results.append((ep_return, ep_cost))
            print(f"Episode {len(results)}: return = {ep_return:.2f}, cost = {ep_cost:.0f}")
            ep_return, ep_cost = 0.0, 0.0

    returns, costs = zip(*results)
    print(f"Mean return {np.mean(returns):.2f} | mean cost {np.mean(costs):.1f}")
    venv.close()


if __name__ == "__main__":
    inspect_env()
    visualize_environment()
    path = train(total_timesteps=200_000)
    evaluate(path)
