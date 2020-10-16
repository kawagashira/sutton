#!/usr/bin/env python
#
#                                   blackjack.py
#
#   Exercise 5.1: Blackjack

import numpy as np


class Env:

    def __init__(self):

        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]

    def initialize(self, player, dealer):
        """
Returns:    state = (player's sum, dealer's showing card, player's usable ace)
        """

        player.draw_card(self.hit())
        player.draw_card(self.hit())
        dealer.draw_card(self.hit())
        dealer.draw_card(self.hit())
        self.showing_card = dealer.cards[0]
        self.state = (player.sum, self.showing_card, player.usable_ace)
        #print ('FIRST 2 HITS: sum', player.sum, 'cards', player.cards, player.usable_ace)
        return self.state

    def generate_experience(self, player, dealer):

        episode = []
        self.initialize(player, dealer)
        """
        print ('Epi:%d' % (e+1), 'player %8s %2d' % (player.cards, player.sum),  \
            'dealer %8s %2d' % (dealer.cards, dealer.sum),          \
            env.state)
        """
        i = 4
        ### PLAYER"S TURN ###
        terminated = 0
        #while player.status == 'playing':
        while not terminated:
            i += 1
            old_state = self.state
            action = player.get_policy(old_state)
            if action:          # HITS
                card = self.hit()
                player.draw_card(card)
            else:
                card = 0
            reward, terminated = self.update(player, dealer)
            #print (i, 'P', old_state, 'A:card %d,%2d' % (action, card), 'R %+d' % reward, 'sum', player.sum, player.status)
            episode.append([old_state, action, reward])
            if terminated:
                break
            if action == 0:     # STICKS
                break
        #print ('P status', player.status, dealer.status, 'terminated', terminated)

        ### DEALER'S TURN ###
        #while dealer.status == 'playing':
        while not terminated:
            i += 1
            old_state = self.state
            action = dealer.dealer_action()
            if action:          # HITS
                card = self.hit()
                dealer.draw_card(card)
            else:
                card = 0
            reward, terminated = self.update(player, dealer) 
            if action == 0:     # When sticks
                break

        ### COMPARE ###
        if not terminated:
            final_reward = np.sign(player.sum - dealer.sum)
            episode[-1][-1] = final_reward
        return episode

    def hit(self):

        import random
        return self.deck[random.randint(0, 12)]

    def update(self, player, dealer):

        """
Returns reward & terminated
        """
        self.state = [player.sum, self.showing_card, player.usable_ace]
        if player.status == 'natural':
            return +1, 1
        elif dealer.status == 'natural':
            return -1, 1
        elif player.status == 'bust':
            return -1, 1
        elif dealer.status == 'bust':
            return +1, 1
        else:
            return 0, 0


class Agent:

    def __init__(self):

        self.cards = []
        self.sum = 0
        self.usable_ace = False
        self.value = 2 * (np.random.rand(10,10,2) - 0.5)

    def __getitem__(self, state):

        # [player'sum, dealer's showing card, player's uable ace]
        return self.value[state[0]-12, state[1]-1, int(state[2])]

    def get_policy(self, s):
        """
s = (players_sum, dealers_showing_card, player_usable_ace)
        """
        return int((self[s[0], s[1], s[2]]) >=0)

    def dealer_action(self):

        return (self.sum <= 16)      # HITS/STICKS

    def draw_card(self, x):

        self.cards.append(x)
        self.usable_ace = False
        if x == 1:
            ace_sum       = self.sum + 1
            eleven_sum    = self.sum + 11
            if ace_sum == 21:
                self.usable_ace = True
                self.sum = ace_sum
                self.status = 'natural'
            elif eleven_sum == 21:
                self.sum = eleven_sum
                self.usable_ace = False
                self.status = 'natural'
            elif eleven_sum <=20:    # Ace=11
                self.usable_ace = False
                self.sum = eleven_sum
                self.status = 'playing'
            elif ace_sum <=20:       # Ace=1
                self.usable_ace = True
                self.sum = ace_sum
                self.status = 'playing'
            else:
                self.sum = ace_sum
                self.status = 'bust'
        else:       # execpt an ace
            self.sum += x
            if self.sum == 21:
                self.status = 'natural'
            elif self.sum <= 20:
                self.status = 'playing'
            else:
                self.status = 'bust'
        return self.status


class Returns:

    def __init__(self):

        self.data = {}

    def __getitem__(self, s):

        s = tuple(s)
        if not s in self.data.keys():
            self.data[s] = []
        return self.data[s]

    def __str__(self):

        return 'Returns'


GAMMA = 1.0
def main():

    from statistics import mean
    import copy
    returns = Returns()
    env = Env()
    for e in range(1000000):
        player = Agent()
        dealer = Agent()
        episode = env.generate_experience(player, dealer)

        G = 0
        state_list = []
        for i in range(-1, -(len(episode)+1), -1):
            #print (i, episode[i])
            state, action, reward = episode[i]
            if state[0] <= 11:
                continue
            G = GAMMA * G + reward 
            state_list.append(tuple(state))
            if not state in state_list:
                returns[state].append(G)
                old_value = copy.copy(player[state])
                print ('orig', player.value[state[0]-12, state[1]-1, int(state[2])])
                #player.value[state[0]-12, state[1]-1, int(state[2])] = mean(returns[state])
                print ('old: %+.4f new: %+.4f diff %+.4f' %     \
                    (old_value, player[state], player[state] - old_value),  \
                    tuple(state), returns[state])
            #print (returns.data)

main()

