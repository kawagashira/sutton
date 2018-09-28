#!/Users/uu123868/.pyenv/versions/anaconda3-4.1.0/bin/python
#
#                            gridworld.py
#

import random
import numpy as np

class Env:

    def __init__(self):

        random.seed(0)
        self.dim = (5, 5)

    def reset(self):

        self.state = [random.randint(1,4), random.randint(1,4)]

    """
move    1:north, 2:east, 3:south, 4:west        
    """
    def act(self, move):

        if move == 1:        # NORTH
            x, y = 0, -1
        elif move == 2:        # EAST
            x, y = 1, 0
        elif move == 3:        # SOUTH
            x, y = 0, 1
        elif move == 4:        # WEST
            x, y = -1, 0
        prev_state = self.state.copy()
        self.state[0] += x
        self.state[1] += y
        if prev_state == [0,1]:            # A
            self.state = [4,1]            # to A'
            r = +10
        elif prev_state == [0,3]:        # B
            self.state = [2,3]            # to B'
            r = +5
        elif self.state[0] < 0 or self.state[0] >= self.dim[0] or self.state[1] < 0 or self.state[1] >= self.dim[1]:
            self.state = prev_state
            r = -1
        else:
            r = 0
        return r

class Agent:

    def __init__(self, dim):

        self.value = np.zeros(dim)
        print (self.value)

    def random_act(self):

        return random.randint(1, 4)
        
###
if __name__ == '__main__':

    env = Env()
    env.reset()
    agent = Agent(env.dim)
    gamma = 0.9
    for n in range(40):
        print ('repeat', n)
        prev_value = agent.value.copy()
        for i in range(env.dim[0]):
            for j in range(env.dim[1]):
                prev0 = agent.value[i,j]
                value = 0
                for move in range(1, 5):
                    env.state = [i,j]
                    r = env.act(move)
                    value += 0.25 * (r + gamma * prev_value[env.state[0], env.state[1]]) 
                agent.value[i,j] = value
        print (agent.value)

