#!/usr/bin/env python
#
#                            windyworld.py
#

import random
import numpy as np
import pandas as pd
import smooth

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
        #print (self.wind[state0[0]])
        #self.state[1] += y
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
    r = -1
    R = 0
    i = 0
    is_goal = False
    #while r != 0:        # AN EPISODE
    while not is_goal:        # AN EPISODE
        i += 1
        s0 = env.state.copy()
        r, is_goal = env.act(a)
        R += r
        a1 = agent.e_greedy(env.state)
        #print ('Q(s,a)', agent.get_q(s0, a))
        value = agent.get_q(s0, a)
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.get_q(env.state, a1) - agent.get_q(s0, a))
        a = a1
        if is_goal:
            break
    return i, R

def q_learn(env, agent, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    r = -1
    is_goal = False
    #while r != 0:
    while not is_goal:
        i += 1
        s0 = env.state.copy()
        a = ql_agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        R += r
        value = agent.q[s0[0], s0[1], a]
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.max_q(env.state) - agent.get_q(s0, a))
        #if r == 0:
        if is_goal:
            break
    return i, R
        
def actor_critic(env, agent, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    r = -1
    is_goal = False
    #while r != 0:
    while not is_goal:
        i += 1
        s0 = env.state.copy()
        a = agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        R += r
        value = agent.value[s0[0], s0[1]]
        policy = agent.policy[s0[0], s0[1], a]
        #env.map[s0[0], s0[1]] = a
        delta = r + gamma * agent.value[env.state[0], env.state[1]] - value
        #print (i, s0, a, delta, env.state, value, agent.value[env.state[0], env.state[1]], policy, agent.policy[env.state[0], env.state[1], a])
        #print (s0, value, agent.value[s0[0], s0[1]], delta, policy, agent.policy[s0[0], s0[1], a])
        agent.value[s0[0], s0[1]] += delta# * alpha
        agent.policy[s0[0], s0[1], a] += delta# * alpha
        if is_goal:
            break
    env.show_map()
    print (i, R)
    return i, R
###
if __name__ == '__main__':

    epsylon     = 0.1
    alpha       = 0.1
    gamma       = 1.0
    dim         = (10, 7)
    num         = 500

    env = Env()
    agent = Agent(epsylon)
    ql_agent = Agent(epsylon)
    ac_agent    = ActorCriticAgent(dim, epsylon)
    w = []
    for n in range(num):

        s_step, s_r     = sarsa(env, agent, alpha, gamma)
        ql_step, ql_r   = q_learn(env, ql_agent, alpha, gamma)
        ac_step, ac_r   = actor_critic(env, ac_agent, alpha, gamma)
        w.append([n + 1, s_step, s_r, ql_step, ql_r, ac_step, ac_r])

    df = pd.DataFrame(w, columns = ['episode_id', 's_step', 's_r', 'ql_step', 'ql_r', 'ac_step', 'ac_r'])
    df['smoothed_s_r'] = smooth.smooth(np.array(df['s_r']), window_len=20)[:num]
    df['smoothed_ql_r'] = smooth.smooth(np.array(df['ql_r']), window_len=20)[:num]
    df['smoothed_ac_r'] = smooth.smooth(np.array(df['ac_r']), window_len=20)[:num]
    #print (df)
    df.to_excel('windyworld.xlsx', index=False)
    #df.to_csv('windyworld.csv', index=False)
