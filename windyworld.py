#!/usr/bin/env python
#
#                            windyworld.py
#

import os
import shutil
import random
import numpy as np
import pandas as pd
import smooth
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import seaborn as sns
import matplotlib.pyplot as plt

DIRECTION   = {0: 'U', 1:'R', 2:'D', 3:'L'}
ARROW       = {0: '^', 1:'>', 2:'v', 3:'<'}

class Env:

    def __init__(self):

        random.seed(0)
        self.dim = (10, 7)
        self.start  = [0, 3]
        self.goal   = [7, 3]
        self.wind   = [0,0,0,1,1,1,2,2,1,0]

    def reset(self):

        self.state = self.start.copy()
        self.map    = np.full(self.dim, 9)

    """
move    0:north, 1:east, 2:south, 3:west        
    """
    def act(self, move):

        ### set action ID to map ###
        state0 = self.state.copy()
        self.map[state0[0], state0[1]] = move

        if move == 0:           # NORTH
            x, y = 0, 1
        elif move == 1:         # EAST
            x, y = 1, 0
        elif move == 2:         # SOUTH
            x, y = 0, -1
        elif move == 3:         # WEST
            x, y = -1, 0
        self.state[0] += x
        self.state[1] += y + self.wind[state0[0]]   # plus wind
        r = -1
        is_goal = False
        if self.state == self.goal:
            is_goal = True
            r = 0
        if self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] >= self.dim[0]:
            self.state[0] = self.dim[0] - 1
        if self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] >= self.dim[1]:
            self.state[1] = self.dim[1] - 1
        return r, is_goal

    def show_map(self):

        print (np.flipud(np.transpose(env.map)))

class Agent:

    def __init__(self, epsylon):

        self.q = np.zeros((10, 7, 4))
        self.epsylon = epsylon

    def e_greedy(self, state):

        if random.random() < self.epsylon:     # RANDOM
            return random.randint(0, 3)
        else:
            return np.argmax(self.q[state[0], state[1], :])

    def max_q(self, state):

        return max(self.q[state[0], state[1], :])

    def __getitem__(self, s, a):

        return self.q[s[0], s[1], a]

    def get_q(self, s, a):

        return self.q[s[0], s[1], a]

    def show_value(self, png_file):

        m = np.max(self.q, axis=2)
        sns.heatmap(m.transpose())
        plt.savefig(png_file)
        plt.close('all')

    def show_arrow(self):

        m = np.argmax(self.q, axis=2)
        arrow = list(map(lambda x: ''.join([ARROW[x] for x in x]), m.transpose()))
        for a in reversed(arrow):
            print (a)

class ActorCriticAgent:

    def __init__(self, dim, epsylon):

        self.epsylon = epsylon
        self.value = np.zeros(dim)
        self.policy = np.zeros((dim[0], dim[1], 4))

    def e_greedy(self, state):

        if random.random() < self.epsylon:     # RANDOM
            return random.randint(0, 3)
        else:
            return np.argmax(self.policy[state[0], state[1], :])

def sarsa(env, agent, alpha, gamma):

    ### SARSA ###
    env.reset()
    a = agent.e_greedy(env.state)
    a_list = []
    r = -1
    R = 0
    i = 0
    is_goal = False
    while not is_goal:        # AN EPISODE
        a_list.append(a)
        i += 1
        s0 = env.state.copy()
        r, is_goal = env.act(a)
        R += r
        a1 = agent.e_greedy(env.state)
        value = agent.get_q(s0, a)
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.get_q(env.state, a1) - agent.get_q(s0, a))
        a = a1
    a_list.append(a)
    return i, R, ''.join([DIRECTION[a] for a in a_list])

def q_learn(env, agent, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    r = -1
    is_goal = False
    while not is_goal:
        i += 1
        s0 = env.state.copy()
        a = ql_agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        R += r
        value = agent.q[s0[0], s0[1], a]
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.max_q(env.state) - agent.get_q(s0, a))
    return i, R
        
def actor_critic(env, agent, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    r = -1
    is_goal = False
    while not is_goal:
        i += 1
        s0 = env.state.copy()
        a = agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        R += r
        value = agent.value[s0[0], s0[1]]
        policy = agent.policy[s0[0], s0[1], a]
        delta = r + gamma * agent.value[env.state[0], env.state[1]] - value
        agent.value[s0[0], s0[1]] += delta# * alpha
        agent.policy[s0[0], s0[1], a] += delta# * alpha
        if is_goal:
            break
    return i, R

###
if __name__ == '__main__':

    epsylon     = 0.1
    alpha       = 0.5
    gamma       = 1.0
    dim         = (10, 7)
    num         = 1000
    png_dir     = 'png'

    env = Env()
    agent       = Agent(epsylon)
    ql_agent    = Agent(epsylon)
    ac_agent    = ActorCriticAgent(dim, epsylon)
    w = []
    s_step_list = []

    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)
    os.mkdir(png_dir)

    for n in range(num):
        s_step, s_r, s_a     = sarsa(env, agent, alpha, gamma)
        ql_step, ql_r   = q_learn(env, ql_agent, alpha, gamma)
        ac_step, ac_r   = actor_critic(env, ac_agent, alpha, gamma)
        w.append([n + 1, s_step, s_r, ql_step, ql_r, ac_step, ac_r])
        s_step_list.append(s_step)
        print ('%3d %3d %.2f' % (n+1, s_step, np.array(s_step_list[-10:]).mean()), s_a)
        if (n+1) % 10 == 0:
            png_file = '%s/value-%03d.png' % (png_dir, n+1)
            agent.show_value(png_file)
            agent.show_arrow()
            plt.plot(s_step_list)
            #plt.show()
            step_list_file = '%s/step_list.png' % png_dir
            plt.ylim((0,200))
            plt.savefig(step_list_file)
            plt.close('all')


    """
    df = pd.DataFrame(w, columns = ['episode_id', 's_step', 's_r', 'ql_step', 'ql_r', 'ac_step', 'ac_r'])
    df['smoothed_s_r'] = smooth.smooth(np.array(df['s_r']), window_len=20)[:num]
    df['smoothed_ql_r'] = smooth.smooth(np.array(df['ql_r']), window_len=20)[:num]
    df['smoothed_ac_r'] = smooth.smooth(np.array(df['ac_r']), window_len=20)[:num]
    #df.to_csv('windyworld.csv', index=False)
    """
