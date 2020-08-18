#!/usr/bin/env python
#
#                            windyworld.py
#

import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from datetime import datetime as dt


class Env:

    def __init__(self, agent_type=4):

        self.agent_type = agent_type     # 4 for four move; 8 for king's move
        random.seed(0)
        self.dim = (10, 7)
        self.start  = [0, 3]
        self.goal   = [7, 3]
        self.wind   = [0,0,0,1,1,1,2,2,1,0]
        self.FOURMOVE = {
            0: (0, 1),      # NORTH
            1: (1, 0),      # EAST
            2: (0, -1),     # SOUTH
            3: (-1, 0)}     # WEST
        self.KINGSMOVE = {
            0: (0, 1),      # NORTH
            1: (1, 1),      # NORTHEAST
            2: (1, 0),      # EAST
            3: (1, -1),     # SOUTHEAST
            4: (0, -1),     # SOUTH
            5: (-1, -1),    # SOUTHWEST
            6: (-1, 0),     # WEST
            7: (-1, 1)}     # NORTHWEST

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

        if self.agent_type == 4:
            x, y = self.FOURMOVE[move]
        elif self.agent_type == 8:           # AGENTTYPE == 8
            x, y = self.KINGSMOVE[move]

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


class AbstractAgent:

    def __init__(self, epsilon):

        self.q = np.zeros((10, 7, self.action_size))
        self.epsilon = epsilon

    def e_greedy(self, state):

        if random.random() < self.epsilon:     # RANDOM
            return random.randint(0, self.action_size - 1)
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
        arrow = list(map(lambda x: ' '.join([self.ARROW[x] for x in x]), m.transpose()))
        for a in reversed(arrow):
            print (a)


class FourMoveAgent(AbstractAgent):

    def __init__(self, epsilon):

        self.action_size = 4
        self.q = np.zeros((10, 7, self.action_size))
        self.epsilon = epsilon
        self.AGENTTYPE = 4
        self.DIRECTION   = {0: 'U', 1:'R', 2:'D', 3:'L'}
        self.ARROW       = {0: '^', 1:'>', 2:'v', 3:'<'}
            

class KingsMoveAgent(AbstractAgent):

    def __init__(self, epsilon):

        self.action_size = 8
        self.q = np.zeros((10, 7, self.action_size))
        self.epsilon = epsilon
        self.AGENTTYPE = 8
        #self.DIRECTION = {0: 'U', 1:'r', 2:'R', 3:'e', 4: 'D', 5:'w', 6:'L', 7:'l'}
        self.DIRECTION  = {0: '⬆️', 1:'↗️', 2:'➡️', 3:'↘️', 4: '⬇️', 5:'↙️', 6:'⬅️', 7:'↙️'}
        #self.ARROW     = {0: '^', 1:'/', 2:'>', 3:'\\', 4: 'v', 5:'/', 6:'<', 7:'\\'}
        self.ARROW     = {0: '⬆️', 1:'↗️', 2:'➡️', 3:'↘️', 4: '⬇️', 5:'↙️', 6:'⬅️', 7:'↙️'}
            
            
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
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a)   \
            + alpha * (r + gamma * agent.get_q(env.state, a1) - agent.get_q(s0, a))
        a = copy.copy(a1)
    a_list.append(a)
    return i, R, ' '.join([agent.DIRECTION[a] for a in a_list])


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
        agent.value[s0[0], s0[1]] += delta
        agent.policy[s0[0], s0[1], a] += delta
        if is_goal:
            break
    return i, R


def show_step_graph(step_list, std_list, png_file):

    plt.plot(s_step_list, label='#steps')
    plt.plot(std_list, label='SD')
    plt.yscale('log')
    plt.savefig(png_file)
    plt.close('all')
    return


###
if __name__ == '__main__':

    epsilon     = 0.1
    #alpha       = 0.5
    alpha       = 0.1
    #alpha       = 0.01
    gamma       = 1.0
    dim         = (10, 7)
    num         = 4000
    slide       = 20

    now = dt.now()
    png_dir     = now.strftime('png-%y%m%d-%H%M%S')
    #png_dir     = 'png'

    agent       = KingsMoveAgent(epsilon)
    ql_agent    = KingsMoveAgent(epsilon)
    ac_agent    = ActorCriticAgent(dim, epsilon)
    env         = Env(agent.AGENTTYPE)
    w = []
    s_step_list, step_std_list = [], []

    if os.path.isdir(png_dir):
        shutil.rmtree(png_dir)
    os.mkdir(png_dir)
    step_graph_file = '%s/step_list.png' % png_dir

    for n in range(num):
        s_step, s_r, s_a     = sarsa(env, agent, alpha, gamma)
        #ql_step, ql_r   = q_learn(env, ql_agent, alpha, gamma)
        #ac_step, ac_r   = actor_critic(env, ac_agent, alpha, gamma)
        #w.append([n + 1, s_step, s_r, ql_step, ql_r, ac_step, ac_r])
        w.append([n + 1, s_step, s_r])
        s_step_list.append(s_step)
        step_slide  = np.array(s_step_list[-slide:])
        step_std_list.append(step_slide.mean())
        print ('%3d %3d %2.2f %2.2f' % (n+1, s_step, step_slide.mean(), step_slide.std()), s_a)
        if (n+1) % 10 == 0:
            png_file = '%s/value-%03d.png' % (png_dir, n+1)
            agent.show_value(png_file)
            agent.show_arrow()
            show_step_graph(s_step_list, step_std_list, step_graph_file)
