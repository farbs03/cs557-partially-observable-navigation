"""
Record a video of a trained RecurrentPPO agent in Safety-Gymnasium, with a
side panel showing what its lidar actually reports after the PartialLidar mask.

Left:  MuJoCo render of the environment. (The ring of small spheres above the
       agent is Safety-Gymnasium's built-in lidar display, which always shows
       the FULL lidar, not the masked one.)
Right: Top-down, agent-centered view of the lidar. Forward is up.
       Hollow markers = everything the full lidar detects.
       Filled markers = what the agent receives after masking.
       Shaded wedge = field of view; dashed circle = sensing range.

Requires: pip install imageio-ffmpeg matplotlib   (for .mp4 output)
Usage:    python visualize_agent.py
"""

import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from partial_lidar import make_env, OBS_CONFIG

MODEL_PATH = "rppo_safety_goal"        # same save_path used in train()
VIDEO_PATH = "agent_rollout.mp4"
N_EPISODES = 1
CAMERA = "track"       # "track" follows the agent; also "fixednear", "fixedfar", "vision"
FRAME_SIZE = 480       # pixels (square)
RECORD_EVERY = 2       # record every Nth step to keep the video a reasonable length
FPS = 30
SEED = 1000

COLORS = {
    "goal_lidar": "tab:green",
    "hazards_lidar": "tab:blue",
    "vases_lidar": "tab:cyan",
    "pillars_lidar": "midnightblue",
    "gremlins_lidar": "tab:purple",
}


def lidar_points(obs, sl, max_dist):
    """Convert one lidar block into (angles, distances, strengths) for nonzero bins."""
    readings = obs[sl]
    n = len(readings)
    angles = (np.arange(n) + 0.5) * 2 * np.pi / n    # CCW from forward
    hit = readings > 0
    return angles[hit], (1.0 - readings[hit]) * max_dist, readings[hit]


def draw_panel(fig, ax, wrapper, full_obs, seen_obs, title):
    ax.clear()
    max_dist = wrapper.lidar_max_dist
    ax.set_theta_zero_location("N")     # forward points up
    ax.set_theta_direction(1)           # counter-clockwise, matching the lidar bins
    ax.set_ylim(0, max_dist)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["front", "left", "back", "right"])
    ax.set_yticklabels([])

    # Field of view and sensing range.
    if wrapper.fov_deg < 360:
        half = np.deg2rad(wrapper.fov_deg) / 2
        theta = np.linspace(-half, half, 60)
        ax.fill_between(theta, 0, max_dist, color="gold", alpha=0.15, label="field of view")
    if wrapper.sense_range is not None:
        circle = np.linspace(0, 2 * np.pi, 120)
        ax.plot(circle, np.full_like(circle, wrapper.sense_range), "--",
                color="gray", lw=1, label="sensing range")

    for key, color in COLORS.items():
        if key not in wrapper.slices:
            continue
        sl = wrapper.slices[key]
        a, d, s = lidar_points(full_obs, sl, max_dist)
        ax.scatter(a, d, s=20 + 80 * s, facecolors="none", edgecolors=color, alpha=0.5)
        a, d, s = lidar_points(seen_obs, sl, max_dist)
        ax.scatter(a, d, s=20 + 80 * s, color=color, label=key.replace("_lidar", ""))

    ax.plot(0, 0, "o", color="crimson", ms=8)      # the agent
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.22, 1.22), fontsize=7,
              frameon=False, markerscale=3)

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def record(model_path=MODEL_PATH, video_path=VIDEO_PATH, n_episodes=N_EPISODES):
    venv = DummyVecEnv([make_env(seed=SEED, obs_config=OBS_CONFIG, render_mode="rgb_array",
                                 width=FRAME_SIZE, height=FRAME_SIZE, camera_name=CAMERA)])
    venv = VecNormalize.load(model_path + "_vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    model = RecurrentPPO.load(model_path, env=venv)

    wrapper = venv.venv.envs[0]          # the PartialLidar wrapper
    fig = plt.figure(figsize=(FRAME_SIZE / 100, FRAME_SIZE / 100), dpi=100)
    ax = fig.add_subplot(projection="polar")
    fig.subplots_adjust(top=0.84, bottom=0.08)

    venv.seed(SEED)
    obs = venv.reset()
    lstm_states, episode_starts = None, np.ones((1,), dtype=bool)
    episode, step, ep_return, ep_cost, frames = 1, 0, 0.0, 0.0, []

    while episode <= n_episodes:
        if step % RECORD_EVERY == 0:
            title = (f"Episode {episode}  step {step}\n"
                     f"return {ep_return:.1f}  cost {ep_cost:.0f}")
            panel = draw_panel(fig, ax, wrapper, wrapper.last_full_obs,
                               venv.get_original_obs()[0], title)
            frames.append(np.concatenate([wrapper.render(), panel], axis=1))

        action, lstm_states = model.predict(obs, state=lstm_states,
                                            episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        ep_return += float(rewards[0])
        ep_cost += float(infos[0].get("cost", 0.0))
        episode_starts = dones
        step += 1

        if dones[0]:
            print(f"Episode {episode}: return = {ep_return:.2f}, cost = {ep_cost:.0f}")
            episode, step, ep_return, ep_cost = episode + 1, 0, 0.0, 0.0

    plt.close(fig)
    venv.close()
    try:
        imageio.mimsave(video_path, frames, fps=FPS)
    except Exception as err:      # usually: imageio-ffmpeg not installed
        video_path = video_path.rsplit(".", 1)[0] + ".gif"
        print(f"Could not write mp4 ({err}); saving a GIF instead.")
        imageio.mimsave(video_path, frames[::2], duration=1000 * 2 / FPS)
    print(f"Saved {len(frames)} frames to {video_path}")


if __name__ == "__main__":
    record()