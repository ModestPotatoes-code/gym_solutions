import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    # inititate a q nup array with 64 0 states
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('Taxi.pkl','rb')
        q = pickle.load(f)
        f.close()

    # Assign a leaning rate and discount factor
    leaning_rate_a = 0.9 
    discount_factor_g = 0.9
    epsilon = 1
    # epsilon decay rate 1/0.0001 = 10,000 so will take this many steps to stop picking randomly
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # There are 0 to 63 states one for each square on the grid
        state = env.reset()[0] 
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
                # Actions: 0-left 1-down, 2-right, 3-up
            else:
                action = np.argmax(q[state,:])
            
            new_state,reward,terminated,truncated,_ = env.step(action)

            # upodate the q grid with the leaning rate and discount factors

            if is_training:
                q[state,action] = q[state,action] + leaning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # This will stablise the model.
        if(epsilon==0):
            leaning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('Taxi.png')

    f = open('Taxi.pkl','wb')
    pickle.dump(q,f)
    f.close()

if __name__ == '__main__':
    run(15000)

    run(10, is_training=False, render=True)