import gymnasium as gym
import numpy as np


class UnreliableLidar(gym.Wrapper):
    def __init__(
        self,
        env,
        lidar_key="hazards_lidar",
        ray_dropout=0.10,
        scan_dropout=0.02,
        noise_std=0.02,
        max_outage_steps=10,
    ):
        super().__init__(env)

        task = getattr(self.unwrapped, "task", None)
        if task is not None and getattr(task, "observation_flatten", False):
            task.toggle_observation_space()

        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise TypeError(
                "UnreliableLidar requires a Dict observation space containing "
                f"{lidar_key!r}."
            )
        if lidar_key not in self.observation_space.spaces:
            raise KeyError(
                f"The observation space does not contain lidar key {lidar_key!r}."
            )

        self.lidar_key = lidar_key
        self.ray_dropout = ray_dropout
        self.scan_dropout = scan_dropout
        self.noise_std = noise_std
        self.max_outage_steps = max_outage_steps

        self.outage_steps = 0
        self.previous_scan = None
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        self.outage_steps = 0
        self.previous_scan = None

        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs, info = self.env.reset(**kwargs)
        return self._corrupt_lidar(obs), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 6:
            obs, reward, cost, terminated, truncated, info = result
            return (
                self._corrupt_lidar(obs),
                reward,
                cost,
                terminated,
                truncated,
                info,
            )
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._corrupt_lidar(obs), reward, terminated, truncated, info

        raise ValueError(
            "Expected env.step() to return 5 values (Gymnasium) or 6 values "
            "(Safety-Gymnasium), but received "
            f"{len(result)}."
        )

    def _corrupt_lidar(self, observation):
        obs = dict(observation)
        lidar = np.array(obs[self.lidar_key], copy=True)

        if self.outage_steps > 0:
            self.outage_steps -= 1
            if self.previous_scan is not None:
                lidar = self.previous_scan.copy()
            else:
                lidar[:] = 0.0
        elif self._rng.random() < self.scan_dropout:
            self.outage_steps = self._rng.integers(
                1, self.max_outage_steps + 1
            )
            if self.previous_scan is not None:
                lidar = self.previous_scan.copy()
            else:
                lidar[:] = 0.0
        else:
            dropped = self._rng.random(lidar.shape) < self.ray_dropout
            lidar[dropped] = 0.0
            lidar += self._rng.normal(0.0, self.noise_std, size=lidar.shape)
            lidar = np.clip(lidar, 0.0, 1.0)

        self.previous_scan = lidar.copy()
        obs[self.lidar_key] = lidar
        return obs
