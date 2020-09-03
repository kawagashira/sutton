#!/usr/bin/env python
#
#                            cliff_walking.py
#

import random
import numpy as np
import pandas as pd
#import smooth
import sys
sys.path.append('..')
from windyworld.windyworld import FourMoveAgent, softmax

class Cliff:

    def __init__(self, dim, rewards):

        self.rewards    = rewards
        random.seed(0)
        self.dim = dim
        self.start  = [0, 0]
        self.goal   = [11, 0]

    def act(self, move):

        if move == 0:           # NORTH
            x, y = 0, 1
        elif move == 1:         # EAST
            x, y = 1, 0
        elif move == 2:         # SOUTH
            x, y = 0, -1
        elif move == 3:         # WEST
            x, y = -1, 0
        state0 = self.state.copy()
        self.map[state0[0], state0[1]] = move
        self.state[0] += x
        self.state[1] += y
        r = self.rewards[1]
        is_goal = False
        if self.state == self.goal:
            r = self.rewards[0]      # Success
            is_goal = True
        ### ON THE CLIFF ###
        elif 1 <= self.state[0] and self.state[0] <= 10 and self.state[1] == 0:
            r = self.rewards[-1]    # FAIL
            #r = -100
            self.state = self.start.copy()
        if self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] >= self.dim[0]:
            self.state[0] = self.dim[0] - 1
        if self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] >= self.dim[1]:
            self.state[1] = self.dim[1] - 1
        return r, is_goal

    def reset(self):

        self.state = self.start.copy()
        self.reset_map()

    def reset_map(self):

        self.map    = np.full((self.dim[0],self.dim[1]), 9)

    def show_map(self):

        print (np.flipud(np.transpose(self.map)))

class Env:

    def __init__(self):

        random.seed(0)
        self.dim = (10, 7)
        self.start  = [0, 3]
        self.goal   = [7, 3]
        self.wind   = [0,0,0,1,1,1,2,2,1,0]

    def reset(self):

        self.state = self.start.copy()
        self.map    = np.full((10,7), 9)

    """
move    0:north, 1:east, 2:south, 3:west        
    """
    def act(self, move):

        if move == 0:           # NORTH
            x, y = 0, 1
        elif move == 1:         # EAST
            x, y = 1, 0
        elif move == 2:         # SOUTH
            x, y = 0, -1
        elif move == 3:         # WEST
            x, y = -1, 0
        state0 = self.state.copy()
        self.state[0] += x
        self.state[1] += y + self.wind[state0[0]]   # plus wind
        r = -1
        if self.state == self.goal:
            r = 0
        if self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] >= self.dim[0]:
            self.state[0] = self.dim[0] - 1
        if self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] >= self.dim[1]:
            self.state[1] = self.dim[1] - 1
        return r

class Agent:

    def __init__(self, dim, epsylon):
        """
        dim     (x, y)
        """

        self.dim = dim
        self.q = np.zeros((dim[0], dim[1], 4))
        self.epsylon = epsylon

    def e_greedy(self, state):

        if random.random() < self.epsylon:     # RANDOM
            return random.randint(0, 3)
            #return random.randint(1, 4)
        else:
            return np.argmax(self.q[state[0], state[1], :])

    def max_q(self, state):

        return max(self.q[state[0], state[1], :])

    def __getitem__(self, s, a):

        return self.q[s[0], s[1], a]

    def get_q(self, s, a):

        return self.q[s[0], s[1], a]

    def show_policy(self):

        m = ([[np.argmax(self.q[i,j,:]) for i in range(self.dim[0])] for j in range(self.dim[1])])
        print (np.flipud(np.array(m)))

    def show_max_q(self):

        m = ([[np.max(self.q[i,j,:])
                for i in range(self.dim[0])] for j in range(self.dim[1])])
        print (np.flipud(np.array(m)))


class ActorCriticAgent:

    def __init__(self, dim, epsylon):

        self.dim    = dim
        self.epsylon = epsylon
        self.value = np.zeros(dim)
        self.policy = np.zeros((dim[0], dim[1], 4))

    def e_greedy(self, state):

        if random.random() < self.epsylon:     # RANDOM
            a = random.randint(0, 3)
            #print ('random', a, np.argmax(self.policy[state[0], state[1], :]))
            return a
        else:
            return np.argmax(self.policy[state[0], state[1], :])

    def find_policy(self):

        m = [[np.argmax(self.policy[i,j]) for i in range(self.dim[0])] for j in range(self.dim[1])]
        #print (np.flipud(np.array(m)))


def sarsa(env, agent, rewards, alpha, gamma):
    """
    rewards     [goal, transition, fail]
    """
    ### SARSA ###
    env.reset()
    a = agent.e_greedy(env.state)
    r = rewards[1]
    R = 0
    i = 0
    is_goal = False
    while not is_goal:        # AN EPISODE
        i += 1
        s0 = env.state.copy()
        r, is_goal = env.act(a)
        R += r
        if r == rewards[-1]:    # if off the cliff
            env.reset_map()
        else:
            env.map[s0[0], s0[1]] = a
        a1 = agent.e_greedy(env.state)
        #value = agent.get_q(s0, a)
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a)   \
            + alpha * (r + gamma * agent.get_q(env.state, a1) - agent.get_q(s0, a))
        a = a1
        if is_goal or R < -10000:
            break
    return i, R


def expected_sarsa(env, agent, rewards, alpha, gamma):
    """
    Expected Sarsa Algorithm
    rewards     [goal, transition, fail]
    """
    ### SARSA ###
    env.reset()
    a = agent.e_greedy(env.state)
    r = rewards[1]
    R = 0
    i = 0
    is_goal = False
    while not is_goal:        # AN EPISODE
        i += 1
        s0 = env.state.copy()
        r, is_goal = env.act(a)
        R += r
        if r == rewards[-1]:    # if off the cliff
            env.reset_map()
        else:
            env.map[s0[0], s0[1]] = a
        a1 = agent.e_greedy(env.state)
        value = agent.get_q(s0, a)
        #print (np.inner(softmax(agent.q[env.state[0], env.state[1], :]), agent.q[env.state[0], env.state[1], :]))
        target = r + np.inner(
                    softmax(agent.q[env.state[0], env.state[1], :]),
                    agent.q[env.state[0], env.state[1], :])
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) \
                + alpha * (target - agent.get_q(s0, a))
        a = a1
        if is_goal or R < -10000:
            break
    return i, R


def q_learn(env, agent, rewards, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    r = rewards[1]  # transition rewards -1
    is_goal = False
    while not is_goal:
        i += 1
        #print ('q-learn', i, R, agent.epsilon)
        s0 = env.state.copy()
        a = agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        R += r
        if r == rewards[-1]:    # ON THE CLIFF
            env.reset_map()
        else:
            env.map[s0[0], s0[1]] = a
        value = agent.q[s0[0], s0[1], a]
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.max_q(env.state) - agent.get_q(s0, a))
        if is_goal or R < -10000:
            break
    return i, R
        
def actor_critic(env, agent, rewards, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    #r = rewards[1]
    r = -1
    is_goal = False
    agent.find_policy()
    while not is_goal:
        i += 1
        s0 = env.state.copy()
        a = agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        if r == rewards[-1]:
            env.reset_map()
        R += r
        value = agent.value[s0[0], s0[1]]
        policy = agent.policy[s0[0], s0[1], a]
        delta = r + gamma * agent.value[env.state[0], env.state[1]] - value
        agent.value[s0[0], s0[1]] += delta# * alpha
        agent.policy[s0[0], s0[1], a] += delta# * alpha
        if is_goal:
            break
    return i, R


def show_graph(df, png_file):

    import matplotlib.pyplot as plt
    plt.plot(df['smoothed_s_r'],    label='Sarsa')
    plt.plot(df['smoothed_es_r'],   label='Expected Sarsa')
    plt.plot(df['smoothed_ql_r'],   label='Q-Learning')
    #plt.plot(df['s_r'], label='Sarsa')
    #plt.plot(df['ql_r'], label='Q-Learning')
    #plt.plot(df['smoothed_ac_r'], label='Actor-Critic')
    #plt.plot(df['smoothed_gs_r'], label='Greedy Sarsa')
    #plt.plot(df['smoothed_gql_r'], label='Greedy Q-Learning')
    plt.ylim((-140,0))
    plt.legend()
    plt.show()


###
def main(num, epsilon, alpha, gamma, smooth_size, show_arrow=False):

    dim         = (12, 4)
    rewards     = [0, -1, -100]

    env         = Cliff(dim, rewards)
    agent       = FourMoveAgent(dim, 0.1)
    es_agent    = FourMoveAgent(dim, 0.1)
    ql_agent    = FourMoveAgent(dim, 0.1)
    gagent      = FourMoveAgent(dim, 0)     # Greedy
    gql_agent   = FourMoveAgent(dim, 0)     # Greedy
    ac_agent    = ActorCriticAgent(dim, epsilon)
    w = []
    s_list, es_list, ql_list = [], [], []

    for n in range(num):
        s_step, s_r     = sarsa(env, agent, rewards, alpha, gamma)
        es_step, es_r   = expected_sarsa(env, es_agent, rewards, alpha, gamma)
        ql_step, ql_r   = q_learn(env, ql_agent, rewards, alpha, gamma)
        #gs_step, gs_r   = sarsa(env, gagent, rewards, alpha, gamma)
        #gql_step, gql_r = q_learn(env, gql_agent, rewards, alpha, gamma)
        #ac_step, ac_r   = actor_critic(env, ac_agent, rewards, alpha, gamma)
        s_list.append(s_r)
        es_list.append(es_r)
        ql_list.append(ql_r)
        w.append([n + 1,
            s_step, s_r, es_step, es_r, ql_step, ql_r])
            #gs_step, gs_r, gql_step, gql_r,

        if (n+1) % 100 == 0:
            print (n+1, '%.2f' % alpha, s_r, es_r, ql_r)
            #ac_step, ac_r])

    """
    df = pd.DataFrame(
            w,
            columns = ['episode_id',
                's_step', 's_r', 'es_step', 'es_r', 'ql_step', 'ql_r'])
                #'gs_step', 'gs_r', 'gql_step', 'gql_r',
                #'ac_step', 'ac_r'])
    p = np.array(df['s_step'])
    p1 = smooth.smooth(np.array(df['s_step']))
    df['smoothed_s_r']  = smooth.smooth(np.array(df['s_r']), window_len=smooth_size)[:num]
    df['smoothed_es_r'] = smooth.smooth(np.array(df['es_r']), window_len=smooth_size)[:num]
    df['smoothed_ql_r'] = smooth.smooth(np.array(df['ql_r']), window_len=smooth_size)[:num]
    #df['smoothed_ac_r'] = smooth.smooth(np.array(df['ac_r']), window_len=wl)[:num]
    #df['smoothed_gs_r'] = smooth.smooth(np.array(df['gs_r']), window_len=wl)[:num]
    #df['smoothed_gql_r']= smooth.smooth(np.array(df['gql_r']), window_len=wl)[:num]
    """

    if show_arrow:
        print ('SARSA: LAST POLICY')
        agent.show_arrow()

        print ('EXPECTED SARSA: LAST POLICY')
        es_agent.show_arrow()

        print ('Q_LEARNING: LAST POLICY')
        #ql_agent.find_policy()
        ql_agent.show_arrow()

        """
        print ('GREEDY SARSA: LAST POLICY')
        gagent.show_arrow()

        print ('GREEDY Q_LEARNING: LAST POLICY')
        #ql_agent.find_policy()
        gql_agent.show_arrow()
        """

    """
    print (df)
    png_file = 'graph_cliff_walking.png'
    show_graph(df, png_file)
    """

    #return np.mean(np.array(s_list)), np.mean(np.array(es_list)), np.mean(np.array(ql_list))
    return (
        np.mean(np.array(s_list)),
        np.mean(np.array(es_list)),
        np.mean(np.array(ql_list)))


def run(run_num, num, epsilon, alpha, gamma, smooth_size, show_arrow=False):

    ### INTERIM ###
    w = []
    for i in range(run_num):
        rewards = list(main(num, epsilon, alpha, gamma, smooth_size))
        w.append(rewards)
    mean = list(np.array(w).mean(axis=0))
    mean.insert(0, alpha)
    return mean


def show_mean_reward(rewards, rewards2=None):

    import matplotlib.pyplot as plt
    columns = ['alpha_nr', 's_reward_nr', 'es_reward_nr', 'ql_reward_nr']

    if rewards2 is not None:
        df2 = pd.DataFrame(rewards2, columns = columns)
        print (df2)
        plt.plot(df2['alpha_nr'], df2['s_reward_nr'],    color='b', linestyle='-', label='Asymptotic Sarsa')
        plt.plot(df2['alpha_nr'], df2['es_reward_nr'],   color='r', linestyle='-', label='Asymptotic Expected Sarsa')
        plt.plot(df2['alpha_nr'], df2['ql_reward_nr'],   color='k', linestyle='-', label='Asymptotic Q-Learning')

    df = pd.DataFrame(rewards, columns = columns)
    print (df)
    plt.plot(df['alpha_nr'], df['s_reward_nr'],    color='b', linestyle=':', label='Interim Sarsa')
    plt.plot(df['alpha_nr'], df['es_reward_nr'],   color='r', linestyle=':', label='Interim Expected Sarsa')
    plt.plot(df['alpha_nr'], df['ql_reward_nr'],   color='k', linestyle=':', label='Interim Q-Learning')
    plt.ylim((-140,0))
    plt.legend()
    plt.show()


if __name__ == '__main__':

    epsilon     = 0.1
    gamma       = 1.0
    smooth_size = 10
    interim_run, interim_num = 5000, 100
    asympto_run, asympto_num = 10, 10000

    interim_rewards, asympto_rewards = [], []
    for i in range(10, 105, 5):
        alpha = float(i) / 100   

        ### INTERIM ###
        rewards = run(interim_run, interim_num, epsilon, alpha, gamma, smooth_size)
        interim_rewards.append(rewards)
        print ('INTERIM:', interim_rewards)

        ### ASYMPTOTIC ###
        rewards = run(asympto_run, asympto_num, epsilon, alpha, gamma, smooth_size)
        asympto_rewards.append(rewards)
        print ('ASYMPTO:', asympto_rewards)

    show_mean_reward(interim_rewards, asympto_rewards)

