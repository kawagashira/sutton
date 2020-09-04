#!/usr/bin/env python
#
#                            double_qlearn.py
#

import os
import random
import numpy as np
import pandas as pd
import copy
from datetime import datetime as dt
"""
import shutil
"""

class Env:

    def __init__(self, state_size, action_size=2):

        self.state_size     = state_size
        self.action_size    = action_size
        random.seed(0)

    def reset(self):

        self.state = 1  # start with A

    """
move    0:right, 1:left
    """
    def act(self, move):

        if self.state == 1:     # A
            if move == 0:       # right
                self.state = 0  # TERMINAL
                return 1, 1     # terminated
            elif move == 1:     # left
                self.state = 2  # to B
                return 0, 0
        elif self.state == 2:   # B
            reward = np.random.normal(-0.1, 1)
            self.state = 3
            return reward, 1


class DoubleMDP:

    def __init__(self, state_size, action_size, epsilon, initializer='random'):

        self.state_size = state_size
        self.action_size = action_size
        self.epsilon    = epsilon

        if initializer == 'random':
            self.q = np.random.random((2, self.state_size, self.action_size)) - 0.5
        elif initializer == 'zero':
            self.q = np.zeros((2, self.state_size, self.action_size))

    def e_greedy(self, state):

        if random.random() < self.epsilon:     # RANDOM
            a = random.randint(0, 1)
            return a
        else:
            a = np.argmax(np.sum(self.q[:, state, :], axis=0))
            return (np.argmax(np.sum(self.q[:, state, :], axis=0)))

    def __getitem__(self, x):

        if x[1] is None:
            return 0.0
        else:
            return self.q[x]
            

class SimpleMDP:

    def __init__(self, state_size, action_size, epsilon, initializer='random'):

        self.dim     = state_size
        self.epsilon = epsilon

        if initializer == 'random':
            self.q = np.random.random((state_size, action_size)) - 0.5
        elif initializer == 'zero':
            self.q = np.zeros((state_size, action_size))

    def e_greedy(self, state):

        if random.random() < self.epsilon:     # RANDOM
            return random.randint(0, 1)
        else:
            return np.argmax(self.q[state, :])

    def max_q(self, state):

        return max(self.q[state, :])

    def __getitem__(self, s):

        return self.q[s[0], s[1]]

    def get_q(self, s, a):

        return self.q[s[0], s[1], a]


    def show_value(self, png_file):

        m = np.max(self.q, axis=2)
        sns.heatmap(m.transpose())
        plt.savefig(png_file)
        plt.close('all')

    def show_arrow(self):

        m = np.argmax(self.q, axis=2)
        arrow = list(map(lambda x: ' '.join([self.ARROW[x] for x in x]), m.transpose()))
        for a in reversed(arrow):
            print (a)

    def get_action_str(self, a_list):

        if self.action_size == 4:
            delimiter = ''
        elif self.action_size in [8,9]:
            delimiter = ' '
        return delimiter.join([self.DIRECTION[a] for a in a_list])

    def find_policy(self):

        m = ([[np.argmax(self.q[i,j,:]) for i in range(self.dim[0])] for j in range(self.dim[1])])
        print (np.flipud(np.array(m)))


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x))


def q_learn(env, agent, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    i = 0
    is_goal = False
    actions = []
    while not is_goal:
        i += 1
        s0 = copy.copy(env.state)
        a = ql_agent.e_greedy(env.state)   
        if env.state == 1:
            actions.append(a)
        r, is_goal = env.act(a)
        value = agent[s0, a]
        agent.q[s0, a] = agent[s0, a] +     \
            alpha * (r + gamma * agent.max_q(env.state) - agent[s0, a])
    return actions
        

def double_q_learn(env, agent, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    i = 0
    is_goal = False
    actions = []
    while not is_goal:
        i += 1
        s0 = copy.copy(env.state)
        a = agent.e_greedy(env.state)   
        if env.state == 1:   # When in state A
            actions.append(a)
        r, is_goal = env.act(a)
        if random.random() < 0.5:
            j = 0
        else:
            j = 1
        value = agent.q[j, s0, a]
        #print ('j:%d' % j, 'current state:', s0, 'a:%d' % a,
        #    'next state:', env.state, 'reward: %+.3f' % r)
        a1 = np.argmax(agent[j, env.state, :])
        agent.q[j, s0, a] += alpha * (r + gamma * agent[1-j, env.state, a1] - agent[j, s0, a])
    return actions
        

def show_left_rate(d, png_file):

    import matplotlib
    matplotlib.rcParams['backend'] = 'TkAgg'
    import matplotlib.pyplot as plt

    #plt.plot(df['episode_nr'], df['left_nr'])
    plt.plot(d[:, 0], label='Q-learning')
    plt.plot(d[:, 1], label='Double Q-learning')
    plt.legend(loc='upper right')
    plt.savefig(png_file)
    plt.show()
    plt.close('all')
    return


###
if __name__ == '__main__':

    epsilon     = 0.1
    alpha       = 0.1
    gamma       = 1.0
    state_size  = 4
    action_size = 2
    num         = 300
    repeat      = 100
    stochastic_wind = True

    now = dt.now()


    png_dir     = now.strftime('png-%y%m%d-%H%M%S')
    env         = Env(action_size)
    s_step_list, step_std_list = [], []

    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)
    os.mkdir(png_dir)
    step_graph_file = '%s/step_list.png' % png_dir

    data = []
    for m in range(repeat):
        ql_agent    = SimpleMDP(state_size, action_size, epsilon)
        dql_agent   = DoubleMDP(state_size, action_size, epsilon)
        w = []
        for n in range(num):
            ### Q-LEARNING ###
            q_actions = q_learn(env, ql_agent, alpha, gamma)
            q_left_mean = np.mean(q_actions)

            ### DOUBLE Q-LEARNING ###
            dq_actions = double_q_learn(env, dql_agent, alpha, gamma)
            dq_left_mean = np.mean(dq_actions)

            w.append((q_left_mean, dq_left_mean))

        print ('m', m)
        data.append(w)
    data = np.array(data)
    #print (data)
    d = np.mean(data, axis=0)
    #print (d)
    #df = pd.DataFrame(d)
    #print (df)
    show_left_rate(d, '%s/left.png' % png_dir)
