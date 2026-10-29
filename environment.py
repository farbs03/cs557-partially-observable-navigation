import safety_gymnasium
env = safety_gymnasium.make('SafetyPointGoal0-v0', render_mode='human')
'''
Vision Environment
    env = safety_gymnasium.make('SafetyPointCircle0Vision-v0', render_mode='human')
Keyboard Debug environment
due to the complexity of the agent's inherent dynamics, only partial support for the agent.
    env = safety_gymnasium.make('SafetyPointCircle0Debug-v0', render_mode='human')
'''
obs, info = env.reset()
# Set seeds
# obs, _ = env.reset(seed=0)
terminated, truncated = False, False
ep_ret, ep_cost = 0, 0
for _ in range(1000):
    assert env.observation_space.contains(obs)
    act = env.action_space.sample()
    assert env.action_space.contains(act)
    # modified for Safe RL, added cost
    obs, reward, cost, terminated, truncated, info = env.step(act)
    ep_ret += reward
    ep_cost += cost
    if terminated or truncated:
        observation, info = env.reset()

    env.close()