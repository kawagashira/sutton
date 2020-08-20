#!/usr/bin/env python
#
#                            cliff_walking.py
#

import random
import numpy as np
import pandas as pd
import smooth
from windyworld.windyworld import FourMoveAgent

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
        r = rewards[1]
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

        m = ([[np.max(self.q[i,j,:]) for i in range(self.dim[0])] for j in range(self.dim[1])])
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
            print ('random', a, np.argmax(self.policy[state[0], state[1], :]))
            return a
        else:
            return np.argmax(self.policy[state[0], state[1], :])

    def find_policy(self):

        m = [[np.argmax(self.policy[i,j]) for i in range(self.dim[0])] for j in range(self.dim[1])]
        print (np.flipud(np.array(m)))

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
    #while r != 0:        # AN EPISODE
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
        #print ('Q(s,a)', agent.get_q(s0, a))
        value = agent.get_q(s0, a)
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.get_q(env.state, a1) - agent.get_q(s0, a))
        #print (i, s0, 'take', a, ', observe', r, env.state, value, agent.q[s0[0], s0[1], a])
        a = a1
        if is_goal:
        #if r == 0:
            break
    #env.show_map()
    #print ('GOAL', R)
    return i, R

def q_learn(env, agent, rewards, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    r = rewards[1]  # transition rewards -1
    is_goal = False
    #while r != 0:
    while not is_goal:
        i += 1
        s0 = env.state.copy()
        a = ql_agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        R += r
        if r == rewards[-1]:    # ON THE CLIFF
            #env.show_map()
            env.reset_map()
        else:
            env.map[s0[0], s0[1]] = a
        value = agent.q[s0[0], s0[1], a]
        agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.max_q(env.state) - agent.get_q(s0, a))
        #print (i, s0, 'take', a, ', observe', r, env.state, value, agent.q[s0[0], s0[1], a])
        if is_goal:
            break
    #env.show_map()
    #print (np.flipud(np.transpose(env.map)))
    return i, R
        
def actor_critic(env, agent, rewards, alpha, gamma):

    ### Q-LEARNING ###
    env.reset()
    R = 0
    i = 0
    #r = rewards[1]
    r = -1
    is_goal = False
    print ('POLICY')
    agent.find_policy()
    while not is_goal:
        i += 1
        s0 = env.state.copy()
        a = agent.e_greedy(env.state)   
        r, is_goal = env.act(a)
        if r == rewards[-1]:
            #env.show_map()
            env.reset_map()
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
    #env.show_map()
    print (i, R)
    return i, R
###
if __name__ == '__main__':

    epsilon     = 0.1
    alpha       = 0.1
    gamma       = 1.0
    wl          = 50    # Window Length for Smomothing

    dim = (12, 4); rewards = [0, -1, -100]
    env         = Cliff(dim, rewards)
    agent       = FourMoveAgent(dim, epsilon)
    ql_agent    = FourMoveAgent(dim, epsilon)
    ac_agent    = ActorCriticAgent(dim, epsilon)
    w = []
    #num = 2000
    num = 500
    for n in range(num):
    #for n in range(50):

        s_step, s_r     = sarsa(env, agent, rewards, alpha, gamma)
        ql_step, ql_r   = q_learn(env, ql_agent, rewards, alpha, gamma)
        ac_step, ac_r   = actor_critic(env, ac_agent, rewards, alpha, gamma)
        #print (n + 1, ql_step, ql_r)
        w.append([n + 1, s_step, s_r, ql_step, ql_r, ac_step, ac_r])


    df = pd.DataFrame(
            w,
            columns = ['episode_id', 's_step', 's_r', 'ql_step', 'ql_r', 'ac_step', 'ac_r'])
    p = np.array(df['s_step'])
    #print (p, len(p)) 
    p1 = smooth.smooth(np.array(df['s_step']))
    #print (p1, len(p1))
    df['smoothed_s_r'] = smooth.smooth(np.array(df['s_r']), window_len=wl)[:num]
    df['smoothed_ql_r'] = smooth.smooth(np.array(df['ql_r']), window_len=wl)[:num]
    df['smoothed_ac_r'] = smooth.smooth(np.array(df['ac_r']), window_len=wl)[:num]
    df.to_excel('cliff_walking.xlsx', index=False)
    print (df)

    print ('SARSA: LAST POLICY')
    agent.find_policy()
    #agent.show_max_q()
    print ('Q_LEARNING: LAST POLICY')
    ql_agent.find_policy()
    #ql_agent.show_max_q()
    print ('ACTOR-CRITIC: LAST POLICY')
    ac_agent.find_policy()
    #df_ql = pd.DataFrame(w_ql)
