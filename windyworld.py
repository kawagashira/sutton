#!/Users/uu123868/.pyenv/versions/anaconda3-4.1.0/bin/python
#
#                            windyworld.py
#

import random
import numpy as np

class Env:

    def __init__(self):

        random.seed(0)
        self.dim = (10, 7)
        self.start  = [0, 3]
        self.goal   = [7, 3]
        self.wind   = [0,0,0,1,1,1,2,2,1,0]

    def reset(self):

        self.state = self.start.copy()

    """
move    0:north, 1:east, 2:south, 3:west        
    """
    def act(self, move):

        if move == 0:           # NORTH
            x, y = 0, -1
        elif move == 1:         # EAST
            x, y = 1, 0
        elif move == 2:         # SOUTH
            x, y = 0, 1
        elif move == 3:         # WEST
            x, y = -1, 0
        state0 = self.state.copy()
        self.state[0] += x
        self.state[1] += y + self.wind[state0[0]]   # plus wind
        #print (self.wind[state0[0]])
        #self.state[1] += y
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

    def __init__(self, epsylon):

        self.q = np.zeros((10, 7, 4))
        self.epsylon = epsylon
        print (self.q)

    def e_greedy(self, state):

        if random.random() < self.epsylon:     # RANDOM
            return random.randint(0, 3)
        else:
            return np.argmax(self.q[state[0], state[1], :])

    def __getitem__(self, s, a):

        return self.q[s[0], s[1], a]

    def get_q(self, s, a):

        return self.q[s[0], s[1], a]
        
        
###
if __name__ == '__main__':

    epsylon    = 0.1
    alpha       = 0.1
    gamma       = 0.9

    env = Env()
    agent = Agent(epsylon)
    for n in range(8000):
        print ('%d Episode' % n)
        env.reset()
        a = agent.e_greedy(env.state)
        r = -1
        while r != 0:        # AN EPISODE
        #for i in range(100):
            s0 = env.state.copy()
            r = env.act(a)
            a1 = agent.e_greedy(env.state)
            #print ('Q(s,a)', agent.get_q(s0, a))
            value = agent.get_q(s0, a)
            agent.q[s0[0], s0[1], a] = agent.get_q(s0, a) + alpha * (r + gamma * agent.get_q(env.state, a1) - agent.get_q(s0, a))
            print (s0, 'take', a, ', observe', r, env.state, value, agent.q[s0[0], s0[1], a])
            a = a1
            if r == 0:
                break
