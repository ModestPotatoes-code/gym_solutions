# Conda environment has been created from following this 
# youtube vid https://www.youtube.com/watch?v=gMgj4pSHLww&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte
# minimal changes
# Newer python env can be used 3.12
# The Atari environments need to be run on different code. See the ale_validations
# main ste




import gymnasium as gym

# Initialise the environment
env = gym.make("", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()


