#!/usr/bin/env python
#
#                           random_walk.py
#
# Example 6.2 Random Walk in RL: An introduction

import numpy as np
import copy


class Environment:

    def __init__(self, state_size=7, center=3):

        self.state_size = state_size
        self.center     = center
        self.state      = self.center

    def step(self, action):
        """
        Returns termination flag as 0 for non-terminal and 0 for terminal and;
            reward 0 or 1
        reword
        """
        self.state = action
        if action == 0:     # terminal state with reward of 0
            return 1, 0
        if action == self.state_size - 1:   # terminal state with reward of 1
            return 1, 1
        else:       # non-terminal
            return 0, 0


class Agent:

    def __init__(self, state_size):

        self.state_size = state_size
        self.v = np.array([0.5 for i in range(state_size)])

    def select_action(self, state, mode='R'):
    
        """
        mode    selecting mode
                'R': random 0.5:0.5
        Returns the greater state id out of the values of (id - 1) or (id + 1) 
        """
        import random
        if mode == 'R':
            if random.random() >= 0.5:
                return state + 1
            else:
                return state - 1
        else:       # Arg Max
            i = np.argmax(np.array([self.v[state-1], self.v[state+1]]))
            return [state-1, state+1][i]

class MonteCarlo:

    def __init__(self, agent):

        self.agent = agent


    def generate_episode(self, select_mode='R'):

        print ('generate an episode')
        env = Environment()
        episode = []
        while env.state not in [0, self.agent.state_size-1]:
            prev_state = env.state
            action = self.agent.select_action(env.state, select_mode)
            env.step(action)
            terminal_f, reward = env.step(action)
            print ('state', prev_state, 'action', action, reward)
            transition = [prev_state, action, reward]
            episode.append(transition)
        return episode


    def train_61(self, rep, alpha, check_point=[1,10,20,50,100]):

        w = []
        for e in range(rep):
            episode = self.generate_episode()
            g = 0.0
            print ('episode no.', e+1)
            new_v = copy.copy(self.agent.v)
            for i in range(len(episode)): 
                state, action, reward = episode[i]
                error = reward - self.agent.v[state]
                new_v[state] = self.agent.v[state] + alpha * error
                #print ('alpha * error', alpha * error)
                print ('update %d' % (i+1), 
                    ' '.join(map(lambda x: '%1.3f' % x, new_v[1:-1])),
                    '<', state, action, reward, '> error %+1.3f' % error)
            self.agent.v = copy.copy(new_v)
            print (new_v[1:-1])
            #if e+1 in check_point:
            if 1:
                w.append(copy.copy(self.agent.v[1:-1]))
        return w


    def train_62(self, rep, alpha, gamma, check_point=[1,10,20,50,100,200]):

        w = []
        for e in range(rep):
            episode = self.generate_episode()
            g = 0.0
            print ('episode no.', e+1)
            new_v = copy.copy(self.agent.v)
            for i in range(len(episode)): 
                state, action, reward = episode[i]
                if i == len(episode):   # TERMINAL
                    error = reward - self.agent.v[state]
                else:
                    error = reward + gamma * self.agent.v[action] - self.agent.v[state]
                new_v[state] = self.agent.v[state] + alpha * error
                print ('update',
                    ' '.join(map(lambda x: '%1.3f' % x, self.agent.v[1:-1])),
                    'SAR', state, action, reward, 'error', error)
            self.agent.v = copy.copy(new_v)
            if e+1 in check_point:
                w.append(copy.copy(self.agent.v[1:-1]))
        return w


def plot_value(w):

    import matplotlib.pyplot as plt
    print (w)
    for i in range(len(w)):
        print (w[i])
        plt.plot(range(1,len(w[i])+1),  w[i])
    #plt.xlim((0,1))
    plt.show()


if __name__ == '__main__':

    agent = Agent(7)
    method = MonteCarlo(agent)
    w = method.train_61(100, 0.1)
    #w = method.train_62(100, 0.1, 0.80)
    plot_value(w)

