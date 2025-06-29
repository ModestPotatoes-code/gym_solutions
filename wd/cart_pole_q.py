import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False):
    
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if(is_training):
        q = np.zeros((len(pos_space) +1, len(vel_space) +1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # init a 20x20x3 array
    else:
        f = open('cart_pole_q.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1 # alpha or learning rate
    discount_factor_g = 0.99 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00001 # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = []
    i = 0

    while(True):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_pos = np.digitize(state[0], pos_space)
        state_vel = np.digitize(state[1], vel_space)
        state_ang = np.digitize(state[2], ang_space)
        state_ang_vel = np.digitize(state[3], ang_vel_space)

        terminated = False          # True when reached goal

        rewards=0

        while(not terminated and rewards < 10000):

            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_pos, state_vel, state_ang, state_ang_vel, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_pos = np.digitize(new_state[0], pos_space)
            new_state_vel = np.digitize(new_state[1], vel_space)
            new_state_ang = np.digitize(new_state[2], ang_space)
            new_state_ang_vel = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_pos, state_vel, state_ang, state_ang_vel, action] = q[state_pos, state_vel, state_ang, state_ang_vel, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_pos, new_state_vel, new_state_ang, new_state_ang_vel,:]) - q[state_pos, state_vel, state_ang, state_ang_vel, action]
                )

            state = new_state
            state_pos = new_state_pos
            state_vel = new_state_vel
            state_ang = new_state_ang
            state_ang_vel = new_state_ang_vel
                                        
            rewards+=reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i}  Rewards: {rewards}')

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if is_training and i%100==0:
            print(f'Episode: {i} Rewards: {rewards} epsilon:{epsilon:0.2f} Mean Rewards: {mean_rewards:0.2f}')
        
        if mean_rewards > 1000:
            break
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        i += 1

    env.close()

    # Save Q table to file
    if is_training:
        f = open('cart_pole_q.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'cart_pole_q.png')

if __name__ == '__main__':
    # run(is_training=True, render=False)

    run(is_training=False, render=True)
    