import gymnasium as gym
import numpy as np


class ManualObstacleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, walls=None, obstacles=None, goal=None, render_mode=None):
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self._pygame = None
        self._screen = None
        self._clock = None
        self._render_closed = False
        self._window_size = 800
        self._world_extent = 2.0

        # Walls: list of (x1, y1, x2, y2) line segments or rectangles.
        self.walls = walls or [
            (-1.0, -1.0, 1.0, -0.9),
            (-1.0, 1.0, 1.0, 0.9),
            (-1.0, -1.0, -0.9, 1.0),
            (1.0, -1.0, 0.9, 1.0),
        ]

        # Obstacles: list of (x, y, radius)
        self.obstacles = obstacles or [
            (0.0, 0.0, 0.25),
            (-0.4, 0.5, 0.2),
            (0.6, -0.5, 0.23),
        ]

        self.goal = goal or np.array([1.5, 1.5], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)

    def _world_to_screen(self, x, y):
        scale = self._window_size / (2 * self._world_extent)
        screen_x = round((x + self._world_extent) * scale)
        screen_y = round((self._world_extent - y) * scale)
        return screen_x, screen_y

    def _initialize_renderer(self):
        import pygame

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode(
            (self._window_size, self._window_size)
        )
        pygame.display.set_caption("Manual Obstacle Environment")
        self._clock = pygame.time.Clock()

    def _draw_lidar(self, lidar_ranges, lidar_angles):
        if (lidar_ranges is None) != (lidar_angles is None):
            raise ValueError("Provide both lidar_ranges and lidar_angles.")
        if lidar_ranges is None:
            return

        ranges = np.asarray(lidar_ranges, dtype=float)
        angles = np.asarray(lidar_angles, dtype=float)
        if ranges.ndim != 1 or angles.ndim != 1 or ranges.shape != angles.shape:
            raise ValueError("Lidar ranges and angles must be 1D arrays of equal size.")
        if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(angles)):
            raise ValueError("Lidar ranges and angles must contain finite values.")
        if np.any(ranges < 0):
            raise ValueError("Lidar ranges must be non-negative distances.")

        origin = self._world_to_screen(*self.agent_pos)
        for distance, angle in zip(ranges, angles):
            endpoint = self._world_to_screen(
                self.agent_pos[0] + distance * np.cos(angle),
                self.agent_pos[1] + distance * np.sin(angle),
            )
            self._pygame.draw.aaline(
                self._screen, (70, 170, 230), origin, endpoint
            )
            self._pygame.draw.circle(self._screen, (30, 120, 200), endpoint, 3)

    def render(self, lidar_ranges=None, lidar_angles=None):
        if self.render_mode != "human" or self._render_closed:
            return

        if self._screen is None:
            self._initialize_renderer()

        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        screen = self._screen
        screen.fill((248, 249, 251))

        grid_color = (225, 228, 233)
        for coordinate in np.arange(-self._world_extent, self._world_extent + 0.01, 0.5):
            start = self._world_to_screen(coordinate, -self._world_extent)
            end = self._world_to_screen(coordinate, self._world_extent)
            pygame.draw.line(screen, grid_color, start, end, 1)
            start = self._world_to_screen(-self._world_extent, coordinate)
            end = self._world_to_screen(self._world_extent, coordinate)
            pygame.draw.line(screen, grid_color, start, end, 1)

        self._draw_lidar(lidar_ranges, lidar_angles)

        for x1, y1, x2, y2 in self.walls:
            top_left = self._world_to_screen(min(x1, x2), max(y1, y2))
            bottom_right = self._world_to_screen(max(x1, x2), min(y1, y2))
            rect = pygame.Rect(
                top_left[0],
                top_left[1],
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
            )
            pygame.draw.rect(screen, (55, 62, 72), rect)

        for x, y, radius in self.obstacles:
            center = self._world_to_screen(x, y)
            pixel_radius = round(
                radius * self._window_size / (2 * self._world_extent)
            )
            pygame.draw.circle(screen, (215, 83, 75), center, pixel_radius)
            pygame.draw.circle(screen, (160, 50, 45), center, pixel_radius, 2)

        goal = self._world_to_screen(*self.goal)
        pygame.draw.circle(screen, (55, 175, 95), goal, 12)
        pygame.draw.circle(screen, (255, 255, 255), goal, 12, 2)

        agent = self._world_to_screen(*self.agent_pos)
        pygame.draw.circle(screen, (55, 115, 220), agent, 10)
        pygame.draw.circle(screen, (255, 255, 255), agent, 10, 2)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        self._render_closed = True
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        return self.agent_pos.copy(), {}

    def _inside_wall(self, x, y):
        for x1, y1, x2, y2 in self.walls:
            left = min(x1, x2)
            right = max(x1, x2)
            bottom = min(y1, y2)
            top = max(y1, y2)
            if left <= x <= right and bottom <= y <= top:
                return True
        return False

    def _inside_obstacle(self, x, y):
        for cx, cy, r in self.obstacles:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                return True
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        new_pos = self.agent_pos + action * 0.1

        if self._inside_wall(new_pos[0], new_pos[1]) or self._inside_obstacle(
            new_pos[0], new_pos[1]
        ):
            reward = -1.0
            cost = 1.0
            terminated = False
            truncated = False
            return self.agent_pos.copy(), reward, cost, terminated, truncated, {}

        self.agent_pos = new_pos
        dist = np.linalg.norm(self.agent_pos - self.goal)

        reward = -0.01
        cost = 0.0
        terminated = False
        truncated = False

        if dist < 0.2:
            reward += 10.0
            terminated = True

        return self.agent_pos.copy(), reward, cost, terminated, truncated, {}


if __name__ == "__main__":
    env = ManualObstacleEnv(
        render_mode="human",
        walls=[
            (-1.0, -1.0, 1.0, -0.9),
            (-1.0, 1.0, 1.0, 0.9),
            (-1.0, -1.0, -0.9, 1.0),
            (1.0, -1.0, 0.9, 1.0),
            (-0.3, -0.2, 0.3, 0.2),
        ],
        obstacles=[
            (0.0, 0.0, 0.25),
            (-0.5, 0.6, 0.18),
            (0.7, -0.5, 0.22),
        ],
    )

    try:
        obs, _ = env.reset(seed=0)
        env.render()
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            env.render()
            print(f"action={action}, pos={obs}, reward={reward}, cost={cost}")
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
