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
        return self.state, moving, abs(moving) * -2     # 2 dolloars for each transfer

    def to_return(self, car):

        old_state = self.state
        self.state = self.state + np.array(car)
        #print ('state', old_state, '+', np.array(car), '=', self.state)
        self.update()
        return

    def to_request(self, car):

        old_state = self.state
        rentable = np.array([min(car[i], self.state[i]) for i in range(2)])
        reward = rentable * 10
        self.state = self.state - rentable  # rented cars are removed out
        """
        print ('state', old_state, '-', np.array(car), '=', self.state, \
            'rentable', rentable)
        """
        self.update()
        return rentable, reward


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


def policy_evaluation(agent, env, theta, gamma):

    delta = 0.0
    rep = 0
    p = Poisson(env.mean_return, env.mean_request)
    for k in range(3):
        rep += 1
        print ('policy evaluation', rep)
        prob_list = []
        ### FOR ALL STATES ###
        for i in range(env.dim[0]):
            for j in range(env.dim[1]):
                env.state = [i, j]
                value = agent.value[i,j]
                action = agent.get_policy([i,j])
                ### SUM OF P<s,a,s'> [R<s,pi(s),s'> + gamma V(s')] ###
                curr_value = p.expected_value(action, agent, env, gamma)
                agent.value[i, j] = curr_value
        delta = max(delta, abs(value - curr_value))
        print ('delta', delta, value, curr_value)
        print (agent.value)


def policy_improvement(agent, env):

    p = Poisson(env.mean_return, env.mean_request)
    print ('policy improvement')
    policy_stable = True
    for i in range(agent.dim[0]):
        for j in range(agent.dim[1]):
            old_action = agent.get_policy([i,j])
            w = []
            for action in range(agent.move_range[0], agent.move_range[1] + 1):
                env.state = [i,j]
                value = p.expected_value(action, agent, env, gamma)
                w.append(value)
            k = np.argmax(w)
            pi_s = k - agent.move_range[1]
            print (i, j, 'action', old_action, '->', pi_s, old_action != pi_s)
            agent.policy[i,j] = pi_s
            if old_action != pi_s:
                policy_stable = False
    return policy_stable


class Poisson:

    def __init__(self, mean_return, mean_request, max_car=5):

        self.max_car = max_car
        self.ret_prob_1 = [poisson(n, mean_return[0]) for n in range(max_car+1)]
        self.ret_prob_2 = [poisson(n, mean_return[1]) for n in range(max_car+1)]
        self.req_prob_1 = [poisson(n, mean_request[0]) for n in range(max_car+1)]
        self.req_prob_2 = [poisson(n, mean_request[1]) for n in range(max_car+1)]
        print (sum(self.ret_prob_1))

    def expected_value(self, action, agent, env, gamma):
    
        old_state = env.state
        next_state, real_action, transfer_cost = env.transfer(action)
        curr_value = 0.0
        for ret_1 in range(self.max_car):
            for ret_2 in range(self.max_car):
                for req_1 in range(self.max_car):
                    for req_2 in range(self.max_car):
                        env.state = old_state
                        env.to_return((ret_1, ret_2))
                        rentable, credit = env.to_request((req_1, req_2))
                        curr_value +=       \
                            self.ret_prob_1[ret_1] * self.ret_prob_2[ret_2] *     \
                            self.req_prob_1[req_1] * self.req_prob_2[req_2] *   \
                            (transfer_cost + sum(credit) + gamma * agent[next_state])
                        """
                    print (old_state, 'a=%d' % action, env.state,    \
                        (returned_1, returned_2),   \
                        (requested_1, requested_2),
                        transfer_cost, 'credit', credit)
                        """
        return curr_value


'''
def expected_value(action, agent, env, gamma):

    max_car = 5
    prob_list = []
    old_state = env.state
    next_state, real_action, transfer_cost = env.transfer(action)
    curr_value = 0.0
    for returned_1 in range(max_car):
        returned_prob_1 = poisson(returned_1, env.mean_return[0])
        for returned_2 in range(max_car):
            returned_prob_2 = poisson(returned_2, env.mean_return[1])
            for requested_1 in range(max_car):
                requested_prob_1 = poisson(requested_1, env.mean_request[0])
                for requested_2 in range(max_car):
                    env.state = old_state
                    requested_prob_2 = poisson(requested_2, env.mean_request[1])
                    env.to_return((returned_1, returned_2))
                    rentable, credit = env.to_request((requested_1, requested_2))
                    curr_value +=       \
                        returned_prob_1 * returned_prob_2 *     \
                        requested_prob_1 * requested_prob_2 *   \
                        (transfer_cost + sum(credit) + gamma * agent[next_state])
                    """
                    print (old_state, 'a=%d' % action, env.state,    \
                        (returned_1, returned_2),   \
                        (requested_1, requested_2),
                        transfer_cost, 'credit', credit)
                    """
        prob_list.append(returned_prob_1)
    #print ('prob_list', sum(prob_list))
    return curr_value
'''

def poisson(n, lambda_):

    return (lambda_ ** n) * exp(-lambda_) / factorial(n)



gamma = 0.9
theta = 0.001
dim = (21,21)
mean_return     = [3,2]
mean_request    = [3,4]
env = Env(dim, mean_return, mean_request)
agent = Agent(dim)
#agent.randomize_policy()
policy_stable = False
for rep in range(10):
    print ('rep', rep+1)
    policy_evaluation(agent, env, theta, gamma)
    policy_stable = policy_improvement(agent, env)
    print (agent.policy)
    if policy_stable:
        break
