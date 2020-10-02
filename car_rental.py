#!/usr/bin/env python
#
#                            carrental.py
#

import random
import numpy as np
#from numpy.random import poisson
from math import exp, factorial

class Env:

    def __init__(self, dim):

        random.seed(0)
        self.dim = dim
        self.request = [3, 4]
        self.returned = [3, 2]

    def reset(self):

        #self.state = [random.randint(0,20), random.randint(0,20)]
        self.state = [10,10]

    def update(self):

        ### CARS NO MORE THAN 20 ###
        if self.state[0] > 20:
            self.state[0] = 20
        if self.state[1] > 20:
            self.state[1] = 20
        if self.state[0] < 0:
            self.state[0] = 0
        if self.state[1] < 0:
            self.state[1] = 0

    def move(self, n):

        if n >= 0:
            moving = min(n, self.state[0])
        elif n < 0:
            moving = -min(-n, self.state[1])
        self.state[0] -= moving
        self.state[1] += moving
        self.update()
        return abs(moving) * -2

    def requested(self, request, i):

        rent = min(request, self.state[i])
        reward = rent * 10
        self.state[i] -= rent
        self.update()
        return rent, reward

    def henkyaku(self, ret, i):

        self.state[i] += ret
        self.update()
        return

def poisson(n, lambda_):

    return (lambda_ ** n) * exp(-lambda_) / factorial(n)


class Agent:

    def __init__(self, dim):

        print (dim)
        self.dim = dim
        self.value = np.zeros(dim, np.float16)
        self.policy = np.zeros(dim, np.int8)

    def random_act(self, min_move, max_move):

        return random.randint(min_move, max_move)


    def init_policy(self, move_range):

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.policy[i,j] = self.random_act(move_range[0], move_range[1])
        print (self.policy)

    def get_policy(self, s):

        return self.policy[s[0], s[1]]

def policy_evaluation(env, agent, mean_requested, mean_returned, dim, move_range, theta, gamma):

    rep = 0
    dim = (10,10)
    max_n = 11
    prob_req1 = [poisson(m, mean_requested[0])  for m in range(max_n + 1)]
    prob_ret1 = [poisson(m, mean_returned[0] )  for m in range(max_n + 1)]
    prob_req2 = [poisson(m, mean_requested[1])  for m in range(max_n + 1)]
    prob_ret2 = [poisson(m, mean_returned[1])   for m in range(max_n + 1)]
    #print (prob_req1)
    #print (prob_ret1)
    #print (prob_req2)
    #print (prob_ret2)
    delta = 0.0
    #while (delta > theta):
    while 1:
        delta = 0.0
        rep += 1
        value = agent.value.copy()
        for i in range(env.dim[0]):
            v_list = []
            for j in range(env.dim[1]):
                v = agent.value[i,j]
                move = agent.get_policy([i,j])
                ### SUM OF P<s,a,s'> [R<s,pi(s),s'> + gamma V(s')] ###
                value_s = 0.0
                s = 0.0
                for m1 in range(max_n + 1):
                    for n1 in range(max_n + 1):
                        for m2 in range(max_n + 1):
                            for n2 in range(max_n + 1):
                                env.state = [i,j]
                                r = env.move(move)
                                env.henkyaku(n1, 0)
                                env.henkyaku(n2, 1)
                                rent1, reward1 = env.requested(m1, 0)
                                rent2, reward2 = env.requested(m2, 1)
                                #if i in [0] and j in [0]:
                                #    print (i,j, 'move', move, 'r', r, m1, n1, m2, n2, 'rent', rent1, rent2, 'reward', reward1, reward2)
                                value_s += reward1 * prob_req1[m1] * prob_ret1[n1] / (dim[0]**2) + reward2 * prob_req2[m2] * prob_ret2[n2] / (dim[1]**2) + prob_req1[m1] * prob_ret1[n1] * prob_req2[m2] * prob_ret2[n2] * (r + gamma * value[env.state[0], env.state[1]])
                                s += prob_req1[m1] * prob_ret1[n1] * prob_req2[m2] * prob_ret2[n2]
                delta = max(delta, abs(v - value_s))
                #print (i, j, 'move', move, 'value', v, 'to value_s', value_s, s)
                agent.value[i,j] = value_s
                v_list.append(value_s)
            print ('rep', rep, 'i:', i, 'mean value_s', np.mean(v_list), np.std(v_list), s)
            #print (v_list)
        print ('rep:', rep, delta, theta) 
        if theta is None: break
        elif delta < theta: break

def policy_improvement(env, agent, move_range):

    print ('policy improvement')
    policy_stable = True
    for i in range(agent.dim[0]):
        for j in range(agent.dim[1]):
            b = agent.get_policy([i,j])
            w = []
            for move in range(move_range[0], move_range[1] + 1):
                env.state = [i,j]
                r = env.move(move)
                w.append(r + gamma * agent.value[env.state[0], env.state[1]])
            k = np.argmax(w)
            pi_s = k - move_range[1]
            #print (i, j, w, pi_s)
            agent.policy[i,j] = pi_s
            if b != pi_s:
                policy_stable = False
    return policy_stable


gamma = 0.9
theta = 0.01
#theta = 1
dim = (21,21)
#move_range = (-10,10)
move_range = (-5,5)
mean_requested  = [3,4]
mean_returned   = [3,2]
env = Env(dim)
env.reset()
agent = Agent(dim)
agent.init_policy(move_range)
loop = 0
for rep in range(2000):
    loop += 1
    policy_evaluation(env, agent, mean_requested, mean_returned, dim, move_range, theta, gamma)
    policy_stable = policy_improvement(env, agent, move_range)
    print (agent.policy)
    print (loop)
    if policy_stable:
        break

