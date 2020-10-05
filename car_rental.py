#!/usr/bin/env python
#
#                            car_rental.py
#

import random
#import math
import numpy as np
from math import exp, factorial

class Env:

    def __init__(self, dim, mean_return, mean_request):

        random.seed(0)
        self.dim = dim
        self.mean_request = [3, 4]
        self.mean_return  = [3, 2]
        self.reset()

    def reset(self):

        self.state = [10,10]
        print ('reset', self.state)

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

    def transfer(self, a):

        old_state = self.state
        moving = 0
        if a > 0:
            moving = min(a, self.state[0])
        elif a < 0:
            moving = -min(-a, self.state[1])
        self.state[0] -= moving
        self.state[1] += moving
        self.update()
        #print ('transfer state', old_state, self.state, 'a', a, 'moving %+d' % moving)
        return self.state, moving, abs(moving) * -2     # 2 dolloars for each transfer

    def to_return(self, car):

        self.state += np.array(car)
        self.update()
        return

    def to_request(self, car):

        rentable = np.array([min(car[i], self.state[i]) for i in range(2)])
        reward = rentable * 10
        self.state -= rentable
        self.update()
        return rentable, reward


def poisson(n, lambda_):

    return (lambda_ ** n) * exp(-lambda_) / factorial(n)


class Agent:

    def __init__(self, dim, move_range=(-5,5)):

        self.dim = dim
        self.move_range = move_range
        self.max_move = 5
        self.value = np.zeros(dim, np.float32)
        action_size = 2 * self.max_move + 1
        self.policy = np.zeros(dim,  np.int8)

    def random_act(self, min_move, max_move):

        return random.randint(min_move, max_move)

    def randomize_policy(self):

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.policy[i,j] = self.random_act(self.move_range[0], self.move_range[1])

    def get_policy(self, s):

        return self.policy[s[0], s[1]]

    def __getitem__(self, s):

        return self.value[s[0], s[1]]


def policy_evaluation(env, agent, theta, gamma):

    delta = 0.0
    rep = 0
    max_car = 5
    for k in range(5):
        rep += 1
        ### FOR ALL STATES ###
        for i in range(env.dim[0]):
            for j in range(env.dim[1]):
                env.state = [i, j]
                value = agent.value[i,j]
                action = agent.get_policy([i,j])
                ### SUM OF P<s,a,s'> [R<s,pi(s),s'> + gamma V(s')] ###
                v_list, r_list = [], []
                prob_list = []
                next_state, real_action, transfer_cost = env.transfer(action)
                curr_value = 0.0
                for returned_1 in range(max_car):
                    for returned_2 in range(max_car):
                        for requested_1 in range(max_car):
                            for requested_2 in range(max_car):
                                returned_prob_1 = poisson(returned_1, env.mean_return[0])
                                returned_prob_2 = poisson(returned_2, env.mean_return[1])
                                requested_prob_1 = poisson(requested_1, env.mean_request[0])
                                requested_prob_2 = poisson(requested_2, env.mean_request[1])
                                env.to_return((returned_1, returned_2))
                                rentable, credit = env.to_request((requested_1, requested_2))
                                #print (rentable, credit)
                                curr_value +=       \
                                    returned_prob_1 * returned_prob_2 *     \
                                    requested_prob_1 * requested_prob_2 *   \
                                    (transfer_cost + sum(credit) + gamma * agent[next_state])
                print ((i, j), 'a=', action, 'cost=', transfer_cost, next_state, (returned_1, returned_2), (requested_1, requested_2), curr_value)
                #print ((i, j), 'a=', action, r, next_state, (returned_prob_1, returned_prob_2), (requested_prob_1, requested_prob_2), curr_value)

                agent.value[i, j] = curr_value
        delta = max(delta, abs(value - curr_value))
        print ('delta', delta, value, curr_value)
        print (agent.value)


def policy_improvement(env, agent):

    print ('policy improvement')
    policy_stable = True
    for i in range(agent.dim[0]):
        for j in range(agent.dim[1]):
            old_action = np.argmax(agent.get_policy([i,j])) - agent.move_range
            w = []
            for move in range(move_range[0], move_range[1] + 1):
                env.state = [i,j]
                r = env.move(move)
                w.append(r + gamma * agent.value[env.state[0], env.state[1]])
            k = np.argmax(w)
            pi_s = k - move_range[1]
            agent.policy[i,j] = pi_s
            if b != pi_s:
                policy_stable = False
    return policy_stable


gamma = 0.9
theta = 0.001
dim = (21,21)
mean_return     = [3,2]
mean_request    = [3,4]
env = Env(dim, mean_return, mean_request)
agent = Agent(dim)
#agent.randomize_policy()
policy_stable = False
for rep in range(3):
    print ('state', env.state)
    policy_evaluation(env, agent, theta, gamma)
    #policy_stable = policy_improvement(env, agent)
    print (agent.policy)
    if policy_stable:
        break

