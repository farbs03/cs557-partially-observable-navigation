"""
Register Safety-Gymnasium Goal environments with a custom number and kind of
objects, arena size, and lidar settings.

Usage:
    from custom_goal_env import register_goal_env
    env_id = register_goal_env("DenseGoal1", hazards=12, pillars=4, extents=2.0)
    env_id = register_goal_env(
        "FixedGoal1",
        hazards=2,
        hazard_positions=[(-1.0, 0.0), (1.0, 0.0)],
        vases=1,
        vase_positions=[(0.0, 1.0)],
        pillars=2,
        pillar_positions=[(0.0, -1.0), (0.5, -1.0)],
    )
    env = safety_gymnasium.make(env_id)      # -> "SafetyPointFixedGoal1-v0"

Call register_goal_env at module level (not inside `if __name__ == "__main__"`),
so the environment is also registered in SubprocVecEnv worker processes.

Object types:
    hazards  - flat circles; walking through them costs 1 per step (non-physical)
    vases    - small boxes the agent can push; cost on contact if vases_constrained
    pillars  - solid cylinders that block movement; cost on contact
    gremlins - small boxes moving in circles; cost on contact
Each type present gets its own lidar block (e.g. "pillars_lidar"), which
PartialLidar picks up automatically.
Positions are (x, y) coordinates in meters; any objects without an explicit
position are placed randomly.
"""

import math

import gymnasium
import safety_gymnasium
from safety_gymnasium import tasks
from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Hazards, Pillars
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0


def _validate_positions(object_name, positions, count):
    if positions is None:
        return []

    try:
        positions = [tuple(position) for position in positions]
    except TypeError as exc:
        raise ValueError(
            f"{object_name}_positions must be a sequence of (x, y) coordinates"
        ) from exc

    if len(positions) > count:
        raise ValueError(
            f"{object_name}_positions has {len(positions)} entries, "
            f"but {object_name} is set to {count}"
        )

    for index, position in enumerate(positions):
        try:
            valid = len(position) == 2 and all(math.isfinite(value) for value in position)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{object_name}_positions[{index}] must contain finite numeric x and y values"
            ) from exc
        if not valid:
            raise ValueError(
                f"{object_name}_positions[{index}] must contain finite numeric x and y values"
            )

    return positions


def register_goal_env(
    name,
    robot="Point",            # Point, Car, Racecar, Doggo, or Ant
    hazards=8,
    vases=1,
    pillars=0,
    gremlins=0,
    vases_constrained=False,
    extents=1.5,              # objects are placed in [-extents, extents]^2
    lidar_bins=16,
    lidar_max_dist=3.0,
    lidar_type="pseudo",      # "pseudo" (sees through objects) or "natural" (ray casting)
    max_episode_steps=1000,
    hazard_positions=None,
    vase_positions=None,
    pillar_positions=None,
):
    """Register a custom Goal task and return its env id.

    `name` must be CamelCase and end in a digit, e.g. "DenseGoal1" or "PillarGoal0";
    Safety-Gymnasium derives the task class name from the env id this way.
    Each positions argument is a sequence of (x, y) coordinates. Its length
    cannot exceed the corresponding object count; remaining objects are placed
    randomly within the arena.
    """
    assert name[0].isupper() and name[-1].isdigit(), "name must look like 'DenseGoal1'"
    hazard_positions = _validate_positions("hazard", hazard_positions, hazards)
    vase_positions = _validate_positions("vase", vase_positions, vases)
    pillar_positions = _validate_positions("pillar", pillar_positions, pillars)
    env_id = f"Safety{robot}{name}-v0"
    class_name = name[:-1] + "Level" + name[-1]   # what Builder will look up

    class CustomGoal(GoalLevel0):
        def __init__(self, config):
            super().__init__(config=config)      # adds the goal
            self.placements_conf.extents = [-extents, -extents, extents, extents]
            if hazards:
                self._add_geoms(Hazards(
                    num=hazards, locations=hazard_positions, keepout=0.18,
                ))
            if pillars:
                self._add_geoms(Pillars(
                    num=pillars, locations=pillar_positions, is_constrained=True,
                ))
            if vases:
                self._add_free_geoms(Vases(
                    num=vases, locations=vase_positions, is_constrained=vases_constrained,
                ))
            if gremlins:
                self._add_mocaps(Gremlins(num=gremlins, travel=0.35, keepout=0.4))
            # Lidar settings must be set here, before the observation space is built.
            self.lidar_conf.num_bins = lidar_bins
            self.lidar_conf.max_dist = lidar_max_dist
            self.lidar_conf.type = lidar_type

    CustomGoal.__name__ = CustomGoal.__qualname__ = class_name
    setattr(tasks, class_name, CustomGoal)

    if env_id not in gymnasium.envs.registry:
        safety_gymnasium.register(
            id=env_id,
            entry_point="safety_gymnasium.builder:Builder",
            kwargs={"config": {"agent_name": robot}, "task_id": env_id},
            max_episode_steps=max_episode_steps,
        )
    return env_id